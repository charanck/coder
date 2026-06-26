from __future__ import annotations

from config import load_config, load_integration_test_config, load_unit_test_config


def test_load_unit_test_config_uses_fast_local_defaults():
    config = load_unit_test_config()

    assert config.planner_model.model_provider == "local"
    assert config.executor_model.model_provider == "local"
    assert config.langfuse.enabled is False
    assert config.planner_tool_call_limit == 5
    assert config.planner_agent_timeout == 15


def test_load_config_uses_unit_profile(monkeypatch):
    monkeypatch.setenv("APP_CONFIG_PROFILE", "unit")

    config = load_config()

    assert config.planner_model.model_name == "unit-test-planner"
    assert config.executor_model.model_name == "unit-test-executor"


def test_load_integration_test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("PLANNER_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("PLANNER_MODEL_NAME", "planner-test-model")
    monkeypatch.setenv("PLANNER_BASE_URL", "https://planner.example/v1")
    monkeypatch.setenv("PLANNER_API_KEY", "planner-key")
    monkeypatch.setenv("EXECUTOR_MODEL_PROVIDER", "google")
    monkeypatch.setenv("EXECUTOR_MODEL_NAME", "executor-test-model")
    monkeypatch.setenv("EXECUTOR_BASE_URL", "https://executor.example/v1")
    monkeypatch.setenv("EXECUTOR_API_KEY", "executor-key")
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example")
    monkeypatch.setenv("PLANNER_TIMEOUT", "123")
    monkeypatch.setenv("PLANNER_STEP_TIMEOUT", "45")
    monkeypatch.setenv("PLANNER_AGENT_TIMEOUT", "67")
    monkeypatch.setenv("PLANNER_TOOL_CALL_LIMIT", "89")
    monkeypatch.setenv("EXECUTOR_TIMEOUT", "135")

    config = load_integration_test_config()

    assert config.planner_model.model_provider == "openai"
    assert config.planner_model.model_name == "planner-test-model"
    assert config.planner_model.base_url == "https://planner.example/v1"
    assert config.planner_model.api_key == "planner-key"
    assert config.executor_model.model_provider == "google"
    assert config.executor_model.model_name == "executor-test-model"
    assert config.executor_model.base_url == "https://executor.example/v1"
    assert config.executor_model.api_key == "executor-key"
    assert config.langfuse.enabled is True
    assert config.langfuse.public_key == "public"
    assert config.langfuse.secret_key == "secret"
    assert config.langfuse.base_url == "https://langfuse.example"
    assert config.planner_timeout == 123
    assert config.planner_step_timeout == 45
    assert config.planner_agent_timeout == 67
    assert config.planner_tool_call_limit == 89
    assert config.executor_timeout == 135