"""Rendering of recommendation results and theme CSS."""
from pathlib import Path

import streamlit as st

import logic

NAMES = {"rag": "RAG", "n8n": "n8n", "crewai": "CrewAI", "autogen": "Autogen"}
LIGHT = {"bg": "#f5f5f5", "panel": "#ffffff", "text": "#001f3f", "user": "#e0f7fa", "bot": "#ffffff", "line": "#dfe6e9", "accent": "#00d1b2", "navy": "#001f3f"}
DARK = {"bg": "#001a1a", "panel": "#00302e", "text": "#f2f7f7", "user": "#0b4a52", "bot": "#00302e", "line": "#0b4a52", "accent": "#00d1b2", "navy": "#001f3f"}


def theme_css(dark: bool) -> str:
    palette = DARK if dark else LIGHT
    variables = ";".join(f"--{k}:{v}" for k, v in palette.items())
    css = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")
    return f"<style>:root{{{variables}}}\n{css}</style>"


def stack_name(stack: list[str]) -> str:
    order = list(NAMES)
    return " + ".join(NAMES.get(k, k) for k in sorted(stack, key=lambda k: order.index(k) if k in order else 99))


def mermaid(code: str, dark: bool) -> None:
    """Draw a Mermaid flowchart with Streamlit's built-in Graphviz renderer (no CDN needed).

    The source is always available underneath, and is shown alone if the chart cannot be translated.
    """
    dot = logic.mermaid_to_dot(code, dark)
    if dot:
        st.graphviz_chart(dot, width="stretch")
    else:
        st.caption("This diagram could not be drawn; here is its source.")
    with st.expander("Diagram source", expanded=dot is None):
        st.code(code, language="text")


def _bullets(items: list[str]) -> None:
    for item in items:
        st.markdown(f"- {item}")


def comparison_rows(ranking: list[dict]) -> list[dict]:
    rows = []
    for s in ranking:
        c = s["criteria"]
        rows.append(
            {
                "Rank": s["rank"],
                "Stack": stack_name(s["stack"]),
                "Score /10": s["weighted_score"],
                "Fit": c["fit_to_requirements"],
                "Cost": c["cost"],
                "Latency": c["latency"],
                "Effort": c["setup_effort"],
                "Long-term": c["long_term_benefit"],
                "$/day": s["estimated_daily_cost_usd"],
                "ms": s["estimated_latency_ms"],
                "Setup h": s["setup_hours"],
                "Limits": "pass" if s["passes"] else "FAIL",
                "Issues": "; ".join(s["flags"]),
            }
        )
    return rows


def understood(msg: dict) -> None:
    """What the manager (AI) took from the user's input, and what it had to assume."""
    p = msg.get("project")
    if not p:
        return
    r, c = p["requirements"], p["constraints"]
    yes = lambda v: "yes" if v else "no"  # noqa: E731
    with st.expander("What I understood from your input"):
        st.markdown(
            f"- **Comparing:** {', '.join(NAMES.get(o, o) for o in p['options_to_compare'])}\n"
            f"- **Channels:** {', '.join(r['channels'])}\n"
            f"- **Answers from documents:** {yes(r['document_upload'])} · **Messages/day:** {r['messages_per_day']} · "
            f"**Person approves replies:** {yes(r['human_approval_of_replies'])} · "
            f"**Multi-step reasoning:** {yes(r['multi_step_reasoning'])}\n"
            f"- **Limits:** ${c['budget_per_day_usd']}/day · {c['max_latency_ms']} ms · recall ≥ {c['min_recall']} · "
            f"skill {c['user_skill']} · {c['time_available_hours']:g} h available"
        )
        if msg.get("intake_assumptions"):
            st.markdown("**Assumptions I made**")
            _bullets(msg["intake_assumptions"])


def result(msg: dict, dark: bool) -> None:
    rec = msg["rec"]
    for w in rec["warnings"]:
        st.warning(w)
    for e in msg.get("errors", []):
        st.error(f"{e['agent_id']} failed and was excluded: {e['error']}")

    tabs = ["Summary", "Comparison", "Design", "Plan", "Risks"]
    if rec["recall_checklist"]:
        tabs.append("Recall check")
    tabs.append("Agent reports")
    panes = dict(zip(tabs, st.tabs(tabs)))

    with panes["Summary"]:
        st.subheader(rec["headline"])
        st.write(rec["rationale"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Cost / day", f"${rec['estimated_daily_cost_usd']}")
        c2.metric("Cost / month", f"${rec['estimated_monthly_cost_usd']}")
        best = rec["ranking"][0]
        c3.metric("Latency (est.)", f"{best['estimated_latency_ms']} ms")
        st.caption(f"Revision {msg.get('revision', 0)}")
        understood(msg)

    with panes["Comparison"]:
        st.dataframe(comparison_rows(rec["ranking"]), hide_index=True, width="stretch")
        st.caption("Scores are 0-10 per criterion. A stack that fails a hard limit ranks below every stack that passes.")

    with panes["Design"]:
        mermaid(rec["simple_design"], dark)
        st.markdown("**How it works**")
        _bullets(rec["how_it_works"])

    with panes["Plan"]:
        st.markdown("**Phased plan**")
        _bullets(rec["phased_plan"])
        st.markdown("**When to upgrade**")
        _bullets(rec["upgrade_triggers"])

    with panes["Risks"]:
        _bullets(rec["risks"])
        with st.expander("Assumptions behind the estimates"):
            _bullets(rec["assumptions"])

    if "Recall check" in panes:
        with panes["Recall check"]:
            for i, step in enumerate(rec["recall_checklist"], 1):
                st.markdown(f"{i}. {step}")

    with panes["Agent reports"]:
        for r in msg.get("reports", []):
            with st.expander(f"{r['framework']} · ${r['estimated_daily_cost_usd']}/day · ~{r['estimated_latency_ms']} ms"):
                st.write(r["summary"])
                fit = r["fit_to_requirements"]
                st.markdown(f"Covers: {', '.join(fit['covered']) or 'nothing'}  \nMissing: {', '.join(fit['missing']) or 'nothing'}")
                mermaid(r["simple_design"], dark)
                _bullets(r["how_it_works"])
                if r["project_specific_notes"]:
                    st.markdown("**Notes for this project**")
                    _bullets(r["project_specific_notes"])
