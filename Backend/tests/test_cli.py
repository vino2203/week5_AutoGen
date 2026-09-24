import json

import run_manager
from tests.conftest import SAMPLES
from tests.fakes import FakeLLM


async def test_cli_prints_the_recommendation(capsys):
    code = await run_manager.main([str(SAMPLES / "whatsapp_gmail_faq.json")], llm=FakeLLM())
    out = capsys.readouterr().out
    assert code == 0 and "## Recommended: n8n + RAG" in out


async def test_cli_json_output_is_the_session(capsys):
    code = await run_manager.main([str(SAMPLES / "description_only.json"), "--json"], llm=FakeLLM())
    assert code == 0 and json.loads(capsys.readouterr().out)["status"] == "awaiting_approval"


async def test_cli_without_a_key_explains_and_exits_2(capsys):
    code = await run_manager.main([str(SAMPLES / "description_only.json")])
    assert code == 2 and "OPENAI_API_KEY" in capsys.readouterr().err
