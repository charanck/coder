from __future__ import annotations

from config import load_config, load_integration_test_config, load_unit_test_config


def test_load_unit_test_config_uses_fast_local_defaults():
    config = load_unit_test_config()

    assert config.model.model_provider == "local"
    assert config.langfuse.enabled is False
    assert config.planner_tool_call_limit == 5
    assert config.planner_agent_timeout == 15


def test_load_config_uses_unit_profile(monkeypatch):
    monkeypatch.setenv("APP_CONFIG_PROFILE", "unit")

    config = load_config()

    assert config.model.model_name == "unit-test-model"


def test_load_integration_test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example")
    monkeypatch.setenv("MODEL_TIMEOUT", "123")
    monkeypatch.setenv("PLANNER_STEP_TIMEOUT", "45")
    monkeypatch.setenv("PLANNER_AGENT_TIMEOUT", "67")
    monkeypatch.setenv("PLANNER_TOOL_CALL_LIMIT", "89")

    config = load_integration_test_config()

    assert config.model.model_provider == "openai"
    assert config.model.model_name == "test-model"
    assert config.model.base_url == "https://example.com/v1"
    assert config.model.api_key == "test-key"
    assert config.langfuse.enabled is True
    assert config.langfuse.public_key == "public"
    assert config.langfuse.secret_key == "secret"
    assert config.langfuse.base_url == "https://langfuse.example"
    assert config.model_timeout == 123
    assert config.planner_step_timeout == 45
    assert config.planner_agent_timeout == 67
    assert config.planner_tool_call_limit == 89