import json
from pathlib import Path

import pytest

from agents.config import load_config
from agents.manager import Manager
from agents.schemas import ProjectInput
from tests.fakes import FakeLLM, standard_brief

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # tests never touch a real model


@pytest.fixture
def fake():
    return FakeLLM()


@pytest.fixture
def manager(fake):
    return Manager(fake, load_config())


@pytest.fixture
def sample_input() -> ProjectInput:
    return ProjectInput.model_validate(json.loads((SAMPLES / "whatsapp_gmail_faq.json").read_text()))


@pytest.fixture
def project(manager, sample_input):
    """A resolved project, as the manager's brief would produce it."""
    return manager._merge(sample_input, standard_brief(), load_config()["default_constraints"])[0]
