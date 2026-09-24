"""Streamlit front end for the framework advisor. Run: streamlit run app.py"""
import html
import os

import streamlit as st

import logic
import render
from api import Api, ApiError

st.set_page_config(page_title="Framework Advisor", page_icon="🧭", layout="wide")

APPROVE = {"approve", "approved", "yes", "ok"}
REJECT = {"reject", "rejected", "no"}


# ------------------------------------------------------------------- state
def init_state() -> None:
    s = st.session_state
    for key, value in logic.FORM_DEFAULTS.items():  # widgets read their value from here, not from `value=`
        s.setdefault(key, value)
    s.setdefault("messages", [])
    s.setdefault("session", None)  # last /chat response
    s.setdefault("action", None)
    s.setdefault("dark", False)
    s.setdefault("debug", False)
    s.setdefault("backend_url", os.getenv("BACKEND_URL", "http://127.0.0.1:8710"))
    s.setdefault("timeout", 120)
    s.setdefault("last_raw", None)
    # form values decided by the chat handler; widget keys can only be changed before the widgets are drawn
    for key, value in (s.pop("_pending_form", None) or {}).items():
        s[key] = value


def status_of(session: dict | None) -> str | None:
    return session["status"] if session else None


# ----------------------------------------------------------------- dialogs
@st.dialog("Advanced options")
def advanced_dialog() -> None:
    s = st.session_state
    st.caption("Model, temperature and token limits are set by the backend (config.yaml / .env).")
    s.debug = st.toggle("Show raw backend JSON", value=s.debug)
    s.backend_url = st.text_input("Backend URL", value=s.backend_url)
    s.timeout = st.number_input("Request timeout (s)", min_value=10, max_value=600, value=int(s.timeout))
    if st.button("Close"):
        st.rerun()


# ----------------------------------------------------------------- sidebar
def sidebar(api: Api) -> None:
    """Settings only. The project itself lives in the main area."""
    with st.sidebar:
        st.markdown("## Framework Advisor")
        top = st.columns([4, 1])
        top[0].toggle("Dark mode", key="dark")
        if top[1].button("⚙️", help="Advanced options"):
            advanced_dialog()

        with st.expander("Thresholds", expanded=True):
            st.slider("Minimum recall", 0.0, 1.0, step=0.01, key="f_recall")
            st.number_input("Maximum latency (ms)", min_value=100, step=100, key="f_latency")
            st.number_input("Budget ($/day)", min_value=0.05, step=0.25, key="f_budget")
            st.selectbox("Your skill level", logic.SKILLS, key="f_skill")
            st.number_input("Time available (hours)", min_value=1.0, step=4.0, key="f_time")
            status = status_of(st.session_state.session)
            if st.button("Re-score", type="primary", width="stretch",
                         disabled=status not in ("awaiting_approval", "rejected"),
                         help="Apply these thresholds and the weights to the existing analysis"):
                st.session_state.action = {"kind": "rescore"}

        with st.expander("Criteria weights"):
            for k, label in logic.WEIGHT_LABELS.items():
                st.slider(label, 0, 100, key=f"w_{k}")
            st.caption("Weights are normalised, so only the ratios matter.")

        agent_status(api)
        model_settings()
        export_buttons()


def agents_in_use(session: dict | None) -> set[str] | None:
    """Agents that took part in the current analysis (None when there is none yet)."""
    if not session or not (session.get("reports") or session.get("errors")):
        return None
    used = {r["agent_id"] for r in session["reports"]} | {e["agent_id"] for e in session["errors"]}
    return used | {"agent_0_manager", "agent_6_recommender"}


def agent_status(api: Api) -> None:
    try:
        info = api.status()
        agents, error = info.get("agents", {}), None
    except ApiError as e:
        info, agents, error = None, {}, str(e)
    st.session_state["_status_info"] = info
    used = agents_in_use(st.session_state.session)
    with st.expander("Agent status", expanded=False):
        if error:
            st.markdown('<div class="agent-row"><span class="dot red"></span>Backend offline</div>', unsafe_allow_html=True)
        for agent_id, name in logic.AGENTS:
            if used is not None and agent_id not in used:  # the log is global, so old runs must not look current
                st.markdown(f'<div class="agent-row"><span class="dot grey"></span><b>{name}</b> · <small>not used in this analysis</small></div>',
                            unsafe_allow_html=True)
                continue
            entry = agents.get(agent_id)
            color = logic.status_color(entry)
            when = f" <small>{entry['timestamp'][11:19]} UTC</small>" if entry else ""
            state = entry["status"] if entry else "not run yet"
            st.markdown(f'<div class="agent-row"><span class="dot {color}"></span><b>{name}</b> · {state}{when}</div>',
                        unsafe_allow_html=True)
            if entry and entry["status"] == "error":
                st.caption(entry["message"])
        st.caption("Last log line per agent for the current analysis, not a live heartbeat." if used is not None
                   else "Last log line per agent from earlier runs; run an analysis to see the current one.")
        if st.button("Refresh", key="refresh_status"):
            st.rerun()


def model_settings() -> None:
    info = st.session_state.get("_status_info") or {}
    with st.expander("Model settings (read-only)"):
        if not info:
            st.write("Backend offline.")
            return
        st.write(f"AI: **{info.get('llm', '?')}**")
        for key in ("model", "temperature", "max_tokens"):
            if key in info:
                st.write(f"{key}: `{info[key]}`")


def export_buttons() -> None:
    msgs, session = st.session_state.messages, st.session_state.session
    with st.expander("Export"):
        c = st.columns(2)
        c[0].download_button("JSON", logic.transcript_json(msgs, session), "advisor_chat.json", "application/json",
                             disabled=not msgs, width="stretch")
        c[1].download_button("Markdown", logic.transcript_markdown(msgs), "advisor_chat.md", "text/markdown",
                             disabled=not msgs, width="stretch")


# ------------------------------------------------------------ project form
def project_card(has_result: bool) -> None:
    """The project description and optional details; collapses once there is a result."""
    box = st.expander("Project details (edit and press Analyze to run again)", expanded=False) if has_result \
        else st.container(border=True)
    with box:
        st.text_area("What should the bot do?", key="f_description", height=90,
                     placeholder="e.g. Answer customers on WhatsApp and Gmail from our FAQ, about 100 messages a day")
        st.caption("Everything below is optional. Leave it open and the advisor reads it from your text or makes a stated assumption.")
        c = st.columns(2)
        c[0].multiselect("Options to compare", list(logic.OPTION_LABELS), key="f_options", format_func=logic.OPTION_LABELS.get,
                         placeholder="All four, or the ones you name")
        c[1].multiselect("Channels (type to add your own)", logic.CHANNEL_CHOICES, key="f_channels", accept_new_options=True,
                         placeholder="Read from your text")
        c = st.columns(4)
        c[0].selectbox("Answers from documents / FAQs?", logic.CHOICES, key="f_docs",
                       help="Documents are not uploaded here. This only tells the advisor whether retrieval (RAG) is needed.")
        c[1].number_input("Messages per day", min_value=0, key="f_msgs", help="0 = let the advisor read or estimate it")
        c[2].selectbox("A person approves replies?", logic.CHOICES, key="f_approval")
        c[3].selectbox("Multi-step reasoning needed?", logic.CHOICES, key="f_multi", help="Several cooperating agents")
        if st.session_state.f_docs != "No":
            with st.expander("Document details (optional)"):
                d = st.columns(2)
                d[0].number_input("Documents (about)", min_value=0, key="f_doc_count")
                d[1].number_input("Total MB (about)", min_value=0.0, key="f_doc_mb")
        if st.button("Analyze", type="primary"):
            st.session_state.action = {"kind": "analyze"}


# ----------------------------------------------------------------- actions
def add(role: str, text: str = "", **extra) -> None:
    st.session_state.messages.append(logic.new_message(role, text, **extra))


def handle_response(resp: dict, intro: str = "") -> None:
    st.session_state.session = resp
    st.session_state.last_raw = resp
    status = resp["status"]
    if status == "needs_input":
        add("assistant", "\n".join(resp["questions"]) + "\n\nType it here or in the box above.")
    elif status == "error":
        add("assistant", "Analysis failed: " + "; ".join(f"{e['agent_id']}: {e['error']}" for e in resp["errors"]))
    elif status == "rejected":
        add("assistant", "Rejected. Change the project or thresholds and press **Analyze** or **Re-score**.")
    else:
        add("assistant", intro, rec=resp["recommendation"], reports=resp["reports"], errors=resp["errors"],
            md=resp["message"], revision=resp["revision"],
            project=resp["project"], intake_assumptions=resp["assumptions"])


def call(fn, *args, **kwargs) -> dict | None:
    try:
        return fn(*args, **kwargs)
    except ApiError as e:
        add("assistant", f"⚠️ {e}")
        return None


def run_action(api: Api) -> None:
    action = st.session_state.action
    if not action:
        return
    st.session_state.action = None
    form = logic.collect_form(st.session_state)
    session = st.session_state.session
    kind = action["kind"]

    if kind == "analyze":
        payload = logic.build_input(form)
        if not action.get("quiet"):  # text typed in chat is already echoed
            add("user", html.escape(payload["project_description"]) or "(no description yet)")
        body = {"input": payload}
        if session and session["status"] in ("needs_input", "awaiting_approval", "rejected"):
            body["session_id"] = session["session_id"]
        with st.spinner("Agents are analyzing…"):
            resp = call(api.chat, **body)
        if resp:
            handle_response(resp, "Here is my suggestion. Approve it, reject it, or change the thresholds and re-score.")

    elif kind == "rescore":
        c = logic.build_constraints(form)
        add("user", f"Re-score: budget ${c['budget_per_day_usd']}/day · latency ≤ {c['max_latency_ms']} ms · "
                    f"recall ≥ {c['min_recall']} · skill {c['user_skill']} · {c['time_available_hours']:g} h")
        body = {"session_id": session["session_id"], "constraints": c}
        if (w := logic.build_weights(form)):
            body["weights"] = w
        with st.spinner("Re-scoring…"):
            resp = call(api.chat, **body)
        if resp:
            handle_response(resp, "Re-scored with your new thresholds.")

    elif kind in ("approve", "reject"):
        add("user", kind)
        with st.spinner("Sending…"):
            resp = call(api.chat, session_id=session["session_id"], message=kind)
        if resp:
            handle_response(resp, "Approved. This is the final recommendation." if kind == "approve" else "")

    st.rerun()  # refresh the sidebar (agent status, export) with the new state


def on_chat_text(text: str) -> None:
    """Chat box: approve/reject, or free text that always moves the analysis forward.

    The text is sent as the project description; the manager (AI) reads channels, volume, documents and so on
    from it, and states an assumption for anything it had to guess.
    """
    s = st.session_state
    word = text.strip().lower()
    status = status_of(s.session)

    if word in APPROVE | REJECT:
        if status == "awaiting_approval":
            s.action = {"kind": "approve" if word in APPROVE else "reject"}
        else:
            add("user", text)
            add("assistant", "There is nothing waiting for approval yet.")
        return

    add("user", text)
    # after a recommendation, extra text refines the current project instead of replacing it
    s["_pending_form"] = {"f_description": f"{s.f_description.strip()} {text}".strip() if status == "awaiting_approval" else text}
    s.action = {"kind": "analyze", "quiet": True}


# --------------------------------------------------------------------- main
def main() -> None:
    init_state()
    st.markdown(render.theme_css(st.session_state.dark), unsafe_allow_html=True)
    api = Api(st.session_state.backend_url, st.session_state.timeout)
    sidebar(api)

    info = st.session_state.get("_status_info") or {}
    if info.get("llm") == "not configured":
        st.error(info.get("problem") or "The backend has no OpenAI key.")
    st.markdown("# Which framework should I use?")
    st.caption("Describe the project, pick the options you are considering, and the agents will compare them.")

    msgs = st.session_state.messages
    status = status_of(st.session_state.session)
    last_result = max((i for i, m in enumerate(msgs) if m.get("rec")), default=-1)
    project_card(has_result=last_result >= 0)
    run_action(api)  # after the widgets, so a click on Analyze in this run is handled in this run

    for i, m in enumerate(msgs):
        with st.chat_message(m["role"]):
            if m["text"]:
                st.markdown(m["text"], unsafe_allow_html=m["role"] == "user")
            if m.get("rec"):
                if i == last_result:
                    render.result(m, st.session_state.dark)
                    if status == "awaiting_approval":
                        b = st.columns([1, 1, 6])
                        if b[0].button("Approve", type="primary", key="btn_approve"):
                            st.session_state.action = {"kind": "approve"}
                            st.rerun()
                        if b[1].button("Reject", key="btn_reject"):
                            st.session_state.action = {"kind": "reject"}
                            st.rerun()
                else:
                    with st.expander(f"Earlier recommendation (revision {m.get('revision', 0)}): {m['rec']['headline']}"):
                        st.markdown(m["md"])
            st.markdown(f'<div class="ts">{m["ts"]}</div>', unsafe_allow_html=True)

    if st.session_state.debug and st.session_state.last_raw:
        with st.expander("Raw backend JSON (debug)"):
            st.json(st.session_state.last_raw)

    if text := st.chat_input("Describe your project, answer a question, or type approve / reject"):
        on_chat_text(text)
        st.rerun()


main()
