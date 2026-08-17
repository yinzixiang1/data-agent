"""Agent-scoped Milvus collection names."""


def agent_collection_name(base_name: str, agent_id: int | None) -> str:
    """Return a fresh-project collection name that is always explicitly scoped."""
    if agent_id is None:
        return f"{base_name}_agent_local"
    if agent_id <= 0:
        raise ValueError("agent_id 必须为正整数")
    return f"{base_name}_agent_{agent_id}"
