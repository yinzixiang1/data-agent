from src.retrieval.agent_config import AgentConfigLoader, AgentRuntimeConfig


def _tool(name: str) -> dict:
    return {
        "name": name,
        "display_name": name,
        "description": "test tool",
        "input_schema": {"type": "object", "properties": {}},
        "binding_config": {},
    }


def test_agent_tool_bindings_are_the_only_runtime_enablement_source() -> None:
    loader = object.__new__(AgentConfigLoader)
    config = AgentRuntimeConfig()

    loader._apply_agent_configs(
        config,
        {
            "tool": {
                "choice": "auto",
                "enabled_tools": ["legacy_tool_that_must_be_ignored"],
            }
        },
        {},
        [_tool("bound_tool")],
    )

    assert [tool["name"] for tool in config.tools] == ["bound_tool"]


def test_tool_policy_none_disables_bound_tools() -> None:
    loader = object.__new__(AgentConfigLoader)
    config = AgentRuntimeConfig()

    loader._apply_agent_configs(
        config,
        {"tool": {"choice": "none"}},
        {},
        [_tool("bound_tool")],
    )

    assert config.tools == []
