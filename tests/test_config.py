import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _reload(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import config
    return importlib.reload(config)


def test_defaults(monkeypatch):
    monkeypatch.delenv("APP_NAME", raising=False)
    config = _reload(monkeypatch)
    assert config.settings.raw_data_dir.name == "raw"
    assert config.settings.processed_data_dir.name == "processed"


def test_env_overrides(monkeypatch):
    config = _reload(monkeypatch, APP_NAME="meera-test", LOG_LEVEL="debug")
    assert config.settings.app_name == "meera-test"
    assert config.settings.log_level == "DEBUG"


def test_loads_from_dotenv_file(tmp_path, monkeypatch):
    from dotenv import load_dotenv

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    f = tmp_path / ".env"
    f.write_text("ANTHROPIC_API_KEY=abc123\n")
    load_dotenv(f, override=True)
    config = importlib.reload(importlib.import_module("config"))
    assert config.settings.anthropic_api_key == "abc123"


def test_expected_folders_exist():
    root = Path(__file__).resolve().parent.parent
    for d in ["data/raw", "data/processed", "ingestion", "retrieval",
              "generation", "api", "ui", "tests"]:
        assert (root / d).is_dir(), d
