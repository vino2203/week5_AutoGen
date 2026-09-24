"""Pure helpers: build backend payloads from the form and format exports. No Streamlit imports."""
import json
import re
import textwrap
from datetime import datetime

CHANNEL_CHOICES = ["gmail", "whatsapp", "telegram", "slack", "sms", "email", "web chat"]
OPTION_LABELS = {"rag": "RAG", "n8n": "n8n", "crewai": "CrewAI", "autogen": "Autogen"}
SKILLS = ["no-code", "low-code", "code"]
WEIGHT_LABELS = {
    "fit_to_requirements": "Fit to requirements",
    "cost": "Cost",
    "setup_effort": "Setup effort",
    "latency": "Latency",
    "long_term_benefit": "Long-term benefit",
}
DEFAULT_WEIGHTS = {"fit_to_requirements": 30, "cost": 25, "setup_effort": 20, "latency": 15, "long_term_benefit": 10}
AGENTS = [
    ("agent_0_manager", "Manager"),
    ("agent_1_rag", "RAG"),
    ("agent_2_n8n", "n8n"),
    ("agent_3_crewai", "CrewAI"),
    ("agent_5_autogen", "Autogen"),
    ("agent_6_recommender", "Recommender"),
]
CHOICES = ["Let the advisor decide", "Yes", "No"]
# widget key -> default value, so the app and tests share one source. Open fields are read from the text by the AI.
FORM_DEFAULTS = {
    "f_description": "",
    "f_options": [],  # empty: only the options named in the text, else all four
    "f_channels": [],
    "f_docs": CHOICES[0],
    "f_doc_count": 0,
    "f_doc_mb": 0.0,
    "f_msgs": 0,  # 0: the advisor reads or estimates it
    "f_approval": CHOICES[0],
    "f_multi": CHOICES[0],
    "f_skill": "low-code",
    "f_time": 40.0,
    "f_recall": 0.9,
    "f_latency": 1000,
    "f_budget": 2.0,
    **{f"w_{k}": v for k, v in DEFAULT_WEIGHTS.items()},
}


def tri(choice: str) -> bool | None:
    """'Let the advisor decide' -> None (left open), Yes/No -> True/False."""
    return {"Yes": True, "No": False}.get(choice)


def collect_form(state) -> dict:
    """Read the form values from a mapping (st.session_state or a plain dict), falling back to defaults."""
    return {k: state.get(k, v) for k, v in FORM_DEFAULTS.items()}


def describe(form: dict, description: str | None = None) -> str:
    """Project description plus the optional document facts, which the backend only sees as text."""
    text = (description if description is not None else form["f_description"]).strip()
    if form["f_docs"] != "No" and (form["f_doc_count"] or form["f_doc_mb"]):
        bits = []
        if form["f_doc_count"]:
            bits.append(f"about {form['f_doc_count']} documents")
        if form["f_doc_mb"]:
            bits.append(f"roughly {form['f_doc_mb']:g} MB in total")
        text += f" (Document set: {', '.join(bits)}.)"
    return text


def build_constraints(form: dict) -> dict:
    return {
        "budget_per_day_usd": float(form["f_budget"]),
        "max_latency_ms": int(form["f_latency"]),
        "min_recall": float(form["f_recall"]),
        "user_skill": form["f_skill"],
        "time_available_hours": float(form["f_time"]),
    }


def build_weights(form: dict) -> dict | None:
    """None when untouched, so the backend records 'default weights' as an assumption."""
    w = {k: float(form[f"w_{k}"]) for k in DEFAULT_WEIGHTS}
    if all(w[k] == DEFAULT_WEIGHTS[k] for k in w):
        return None
    if sum(w.values()) <= 0:
        return None
    return w


def build_input(form: dict, description: str | None = None) -> dict:
    return {
        "project_description": describe(form, description),
        "options_to_compare": list(form["f_options"]),
        "requirements": {
            "channels": list(form["f_channels"]),
            "document_upload": tri(form["f_docs"]),
            "messages_per_day": int(form["f_msgs"]) or None,
            "human_approval_of_replies": tri(form["f_approval"]),
            "multi_step_reasoning": tri(form["f_multi"]),
        },
        "constraints": build_constraints(form),
        "weights": build_weights(form),
    }


def status_color(entry: dict | None) -> str:
    if not entry:
        return "grey"
    return {"completed": "green", "running": "amber", "error": "red"}.get(entry.get("status"), "grey")


def now() -> str:
    return datetime.now().strftime("%H:%M")


def new_message(role: str, text: str = "", **extra) -> dict:
    return {"role": role, "text": text, "ts": now(), **extra}


def transcript_json(messages: list[dict], session: dict | None) -> str:
    return json.dumps({"messages": messages, "session": session}, indent=2, default=str)


def transcript_markdown(messages: list[dict]) -> str:
    parts = ["# Framework advisor conversation"]
    for m in messages:
        who = "You" if m["role"] == "user" else "Advisor"
        body = m.get("md") or m.get("text", "")
        parts.append(f"**{who}** ({m.get('ts', '')})\n\n{body}")
    return "\n\n---\n\n".join(parts) + "\n"


# --------------------------------------------- Mermaid flowchart -> Graphviz DOT
# The backend writes small Mermaid flowcharts. Streamlit can draw Graphviz natively (no CDN, no iframe), so the
# subset used here (nodes with [] () [()] {} labels, --> -.-> ==> --- edges, |labels|) is translated to DOT.
_NODE = re.compile(
    r'\s*([A-Za-z_]\w*)\s*'
    r'(?:(\[\(|\[\[|\[|\(\(|\(|\{)\s*(?:"([^"]*)"|([^"\]\)\}]*?))\s*(?:\)\]|\]\]|\]|\)\)|\)|\})\s*)?'
)
_EDGE = re.compile(r'\s*(-\.->|-\.-|==>|-->|---)\s*(?:\|([^|]*)\|)?')
_SHAPES = {"[(": "cylinder", "{": "diamond", "((": "ellipse"}
_SKIP = ("style", "classdef", "class ", "click", "linkstyle", "subgraph", "end", "%%")


def _dot_label(text: str) -> str:
    wrapped = "\\n".join(textwrap.wrap(text.replace('"', "'"), 26)) or " "
    return wrapped


def mermaid_to_dot(code: str, dark: bool = False) -> str | None:
    """Translate a small Mermaid flowchart to DOT, or return None if it cannot be understood."""
    lines = [ln.strip() for ln in code.strip().splitlines() if ln.strip()]
    if not lines or not re.match(r"(flowchart|graph)\b", lines[0], re.I):
        return None
    m = re.match(r"(?:flowchart|graph)\s+(LR|RL|TB|TD|BT)", lines[0], re.I)
    rankdir = {"TD": "TB"}.get(m.group(1).upper(), m.group(1).upper()) if m else "LR"

    nodes: dict[str, tuple[str, str]] = {}
    edges: list[tuple[str, str, str, str]] = []

    def read_node(text: str, pos: int) -> tuple[str, int] | None:
        nm = _NODE.match(text, pos)
        if not nm:
            return None
        node_id, opener = nm.group(1), nm.group(2)
        label = nm.group(3) if nm.group(3) is not None else (nm.group(4) or "")
        if opener:
            nodes[node_id] = (_SHAPES.get(opener, "box"), label)
        else:
            nodes.setdefault(node_id, ("box", node_id))
        return node_id, nm.end()

    for line in lines[1:]:
        if line.lower().startswith(_SKIP):
            continue
        first = read_node(line, 0)
        if not first:
            continue
        prev, pos = first
        while (em := _EDGE.match(line, pos)):
            nxt = read_node(line, em.end())
            if not nxt:
                return None
            target, pos = nxt
            edges.append((prev, target, em.group(1), (em.group(2) or "").strip()))
            prev = target
    if not edges:
        return None

    fill, line_c, font, edge_c = ("#0b4a52", "#00d1b2", "#f2f7f7", "#9fe8dc") if dark else ("#e0f7fa", "#00d1b2", "#001f3f", "#4a6572")
    out = [
        "digraph G {",
        f'  rankdir={rankdir}; bgcolor="transparent"; nodesep=0.35; ranksep=0.5;',
        f'  node [fontname="Helvetica", fontsize=11, style="rounded,filled", fillcolor="{fill}", color="{line_c}", fontcolor="{font}", margin="0.15,0.08"];',
        f'  edge [fontname="Helvetica", fontsize=10, color="{edge_c}", fontcolor="{edge_c}"];',
    ]
    for node_id, (shape, label) in nodes.items():
        style = "filled" if shape in ("cylinder", "diamond", "ellipse") else "rounded,filled"
        out.append(f'  "{node_id}" [label="{_dot_label(label)}", shape={shape}, style="{style}"];')
    for a, b, op, text in edges:
        attrs = []
        if "." in op:
            attrs.append("style=dashed")
        if text:
            attrs.append(f'label="{text.replace(chr(34), chr(39))}"')
        out.append(f'  "{a}" -> "{b}"' + (f' [{", ".join(attrs)}]' if attrs else "") + ";")
    out.append("}")
    return "\n".join(out)
