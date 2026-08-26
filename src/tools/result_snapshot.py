"""Short-lived snapshots for operations on an already returned query result."""

from __future__ import annotations

import hashlib
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(frozen=True, slots=True)
class ResultSnapshot:
    query_result: dict[str, Any]
    session_id: str
    agent_id: int
    user_id: str
    context_digest: str
    created_at: float


class ResultSnapshotStore:
    """Keep exact query results briefly so result operations never rerun SQL."""

    def __init__(self, *, ttl_seconds: int = 1800, max_size: int = 500) -> None:
        self.ttl_seconds = max(1, ttl_seconds)
        self.max_size = max(1, max_size)
        self._entries: dict[str, ResultSnapshot] = {}
        self._lock = RLock()

    def put(
        self,
        query_result: dict[str, Any],
        *,
        session_id: str,
        agent_id: int,
        user_id: str,
        context_summary: str,
    ) -> str:
        """Store one successful result and return its opaque cache identifier."""
        snapshot_id = f"nl2sql:result_snapshot:{uuid.uuid4().hex}"
        snapshot = ResultSnapshot(
            query_result=deepcopy(query_result),
            session_id=session_id,
            agent_id=agent_id,
            user_id=user_id,
            context_digest=_context_digest(context_summary),
            created_at=time.monotonic(),
        )
        with self._lock:
            self._evict_expired()
            if len(self._entries) >= self.max_size:
                oldest_id = min(
                    self._entries,
                    key=lambda key: self._entries[key].created_at,
                )
                self._entries.pop(oldest_id, None)
            self._entries[snapshot_id] = snapshot
        return snapshot_id

    def get(
        self,
        snapshot_id: str,
        *,
        session_id: str,
        agent_id: int,
        user_id: str,
        context_summary: str,
    ) -> dict[str, Any] | None:
        """Return an owned snapshot only when it matches the active query context."""
        with self._lock:
            self._evict_expired()
            snapshot = self._entries.get(snapshot_id)
            if snapshot is None:
                return None
            if (
                snapshot.session_id != session_id
                or snapshot.agent_id != agent_id
                or snapshot.user_id != user_id
                or snapshot.context_digest != _context_digest(context_summary)
            ):
                return None
            return deepcopy(snapshot.query_result)

    def _evict_expired(self) -> None:
        expires_before = time.monotonic() - self.ttl_seconds
        expired = [
            snapshot_id
            for snapshot_id, snapshot in self._entries.items()
            if snapshot.created_at < expires_before
        ]
        for snapshot_id in expired:
            self._entries.pop(snapshot_id, None)


def _context_digest(context_summary: str) -> str:
    return hashlib.sha256(context_summary.encode()).hexdigest()
