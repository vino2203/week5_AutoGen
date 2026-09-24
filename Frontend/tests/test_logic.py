import json

import logic


def form(**over):
    f = dict(logic.FORM_DEFAULTS)
    f.update(over)
    return f


def test_open_fields_are_sent_as_open_so_the_ai_can_fill_them():
    p = logic.build_input(form(f_description="bot"))
    assert p["options_to_compare"] == []
    assert p["requirements"] == {"channels": [], "document_upload": None, "messages_per_day": None,
                                 "human_approval_of_replies": None, "multi_step_reasoning": None}
    assert p["constraints"] == {
        "budget_per_day_usd": 2.0, "max_latency_ms": 1000, "min_recall": 0.9,
        "user_skill": "low-code", "time_available_hours": 40.0,
    }
    assert p["weights"] is None  # untouched -> the backend records "default weights"


def test_explicit_choices_are_sent():
    p = logic.build_input(form(f_options=["rag"], f_channels=["sms"], f_docs="No", f_msgs=30, f_approval="Yes", f_multi="No"))
    assert p["options_to_compare"] == ["rag"]
    assert p["requirements"] == {"channels": ["sms"], "document_upload": False, "messages_per_day": 30,
                                 "human_approval_of_replies": True, "multi_step_reasoning": False}


def test_tri_state():
    assert [logic.tri(c) for c in logic.CHOICES] == [None, True, False]


def test_custom_weights_are_sent():
    assert logic.build_weights(form(w_cost=60))["cost"] == 60


def test_all_zero_weights_are_not_sent():
    assert logic.build_weights(form(**{f"w_{k}": 0 for k in logic.DEFAULT_WEIGHTS})) is None


def test_document_facts_are_appended_to_description():
    text = logic.describe(form(f_description="bot", f_doc_count=12, f_doc_mb=3.5))
    assert "about 12 documents" in text and "3.5 MB" in text
    assert "Document set" not in logic.describe(form(f_description="bot", f_docs="No", f_doc_count=12))


def test_description_override():
    assert logic.build_input(form(f_description="old"), "new")["project_description"] == "new"


def test_status_color():
    assert logic.status_color(None) == "grey"
    assert logic.status_color({"status": "completed"}) == "green"
    assert logic.status_color({"status": "running"}) == "amber"
    assert logic.status_color({"status": "error"}) == "red"


def test_exports():
    msgs = [logic.new_message("user", "hi"), logic.new_message("assistant", "x", md="## Recommended: n8n")]
    assert "## Recommended: n8n" in logic.transcript_markdown(msgs)
    assert json.loads(logic.transcript_json(msgs, {"a": 1}))["session"] == {"a": 1}


def test_mermaid_to_dot_keeps_parentheses_in_labels_and_all_edges():
    code = (
        "flowchart LR\n"
        '    U["Customer (WhatsApp / Gmail)"] --> N["n8n workflow (triggers + replies)"]\n'
        '    D["Docs"] --> V[("Vector store")]\n'
        "    N --> V\n"
        '    V --> H{"Approve?"} -->|yes| R["Send"]\n'
        "    R -.-> U"
    )
    dot = logic.mermaid_to_dot(code)
    assert 'label="Customer (WhatsApp /\\nGmail)"' in dot or "Customer (WhatsApp" in dot
    assert "WhatsApp / Gmail)" in dot.replace("\\n", " ")
    assert 'shape=cylinder' in dot and 'shape=diamond' in dot
    for edge in ('"U" -> "N"', '"D" -> "V"', '"N" -> "V"', '"V" -> "H"', '"H" -> "R" [label="yes"]', '"R" -> "U" [style=dashed]'):
        assert edge in dot, edge


def test_mermaid_to_dot_rejects_non_flowcharts():
    assert logic.mermaid_to_dot("sequenceDiagram\n A->>B: hi") is None
    assert logic.mermaid_to_dot("flowchart LR\n A") is None  # no edges


def test_all_recorded_diagrams_translate():
    import json
    from pathlib import Path

    r = json.loads((Path(__file__).parent / "fixtures" / "analyzed.json").read_text())
    diagrams = [r["recommendation"]["simple_design"]] + [x["simple_design"] for x in r["reports"]]
    for d in diagrams:
        dot = logic.mermaid_to_dot(d, dark=True)
        assert dot and "->" in dot
        assert "label=\"N\"" not in dot  # a label of just the id means a node definition was missed
