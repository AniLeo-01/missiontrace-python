"""
MissionTrace SDK — Context Propagation

Uses Python contextvars for automatic parent-child action nesting.
Works correctly across async/await boundaries.
"""

from __future__ import annotations

import contextvars
import time
import traceback
import uuid
from typing import Any

from missiontrace.core.models import (
    Action,
    ActionStatus,
    ActionType,
    ErrorInfo,
    Mission,
    MissionStatus,
)

# ---------------------------------------------------------------------------
# Context variables — the core of trace propagation
# ---------------------------------------------------------------------------

_current_mission: contextvars.ContextVar[Mission | None] = contextvars.ContextVar(
    "missiontrace_mission", default=None
)
_current_action: contextvars.ContextVar[Action | None] = contextvars.ContextVar(
    "missiontrace_action", default=None
)


def get_current_mission() -> Mission | None:
    return _current_mission.get(None)


def get_current_action() -> Action | None:
    return _current_action.get(None)


# ---------------------------------------------------------------------------
# TraceContext — manages the lifecycle of missions and actions
# ---------------------------------------------------------------------------

class TraceContext:
    """
    Central trace state manager.  One TraceContext per SDK instance.

    All created actions are appended to an internal buffer.
    The transport layer drains this buffer on a schedule.
    """

    def __init__(self, project_id: str = "") -> None:
        self.project_id = project_id
        self._buffer: list[Action | Mission] = []
        self._action_tokens: dict[str, contextvars.Token] = {}
        self._mission_token: contextvars.Token | None = None

    # -- mission lifecycle --------------------------------------------------

    def start_mission(
        self,
        name: str | None = None,
        trigger: str = "manual",
        metadata: dict[str, Any] | None = None,
    ) -> Mission:
        mission = Mission(
            id=uuid.uuid4().hex,
            project_id=self.project_id,
            name=name,
            trigger=trigger,
            metadata=metadata or {},
        )
        self._mission_token = _current_mission.set(mission)
        self._buffer.append(mission)
        return mission

    def end_mission(
        self, mission: Mission, status: MissionStatus = MissionStatus.COMPLETED
    ) -> None:
        mission.ended_at = time.time()
        mission.status = status
        self._buffer.append(mission)  # Re-add to buffer so completion status gets exported
        if self._mission_token is not None:
            _current_mission.reset(self._mission_token)
            self._mission_token = None

    # -- action lifecycle ---------------------------------------------------

    def start_action(
        self,
        type: ActionType,
        name: str,
        input: dict[str, Any] | None = None,
        intent: str | None = None,
    ) -> Action:
        parent = _current_action.get(None)
        current_mission = _current_mission.get(None)

        action = Action(
            id=uuid.uuid4().hex,
            mission_id=current_mission.id if current_mission else "",
            parent_id=parent.id if parent else None,
            type=type,
            intent=intent,
            name=name,
            input=input or {},
            status=ActionStatus.RUNNING,
            started_at=time.time(),  # Wall-clock time for accurate timestamps
        )
        token = _current_action.set(action)
        self._action_tokens[action.id] = token
        self._buffer.append(action)
        return action

    def end_action(
        self,
        action: Action,
        output: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        action.ended_at = time.time()
        action.duration_ms = (action.ended_at - action.started_at) * 1000
        action.output = output

        if error is not None:
            tb = traceback.format_exception(type(error), error, error.__traceback__)
            # Truncate to last 10 frames
            action.error = ErrorInfo(
                type=type(error).__name__,
                message=str(error),
                traceback="".join(tb[-10:]),
            )
            action.status = ActionStatus.FAILED
        else:
            action.status = ActionStatus.COMPLETED

        # Re-add to buffer so completion status gets exported
        self._buffer.append(action)

        # Restore parent context
        token = self._action_tokens.pop(action.id, None)
        if token is not None:
            _current_action.reset(token)

    # -- buffer access (used by transport) ----------------------------------

    def drain_buffer(self) -> list[Action | Mission]:
        """Atomically swap the buffer for an empty list and return contents."""
        items = self._buffer
        self._buffer = []
        return items

    # -- GitHub context attachment ------------------------------------------

    async def attach_github_context(
        self,
        action: Action,
        repository: str,
        file_paths: list[str],
        branch: str = "main",
        mcp_client: Any = None,
    ) -> None:
        """Attach GitHub repository context to an action.

        Fetches code snippets via GitHub MCP and stores in action metadata.

        Args:
            action: The action to attach context to
            repository: GitHub repository in "owner/repo" format
            file_paths: List of file paths to fetch (e.g., ["src/app.py", "tests/test_app.py"])
            branch: Git branch (default: "main")
            mcp_client: Optional GitHub MCP client (creates one if not provided)

        Example:
            with missiontrace.action("tool_call", name="apply_patch") as act:
                await missiontrace.attach_github_context(
                    act,
                    repository="anthropics/anthropic-sdk-python",
                    file_paths=["src/anthropic/client.py"],
                    branch="main"
                )
        """
        from missiontrace.integrations.github_mcp import GitHubMCPClient

        # Create client if not provided
        client = mcp_client or GitHubMCPClient()
        should_close = mcp_client is None

        try:
            # Fetch repository info
            repo_info = await client.fetch_repository_info(repository, branch)

            # Fetch file contents
            files = []
            for file_path in file_paths:
                try:
                    file_data = await client.fetch_file_content(
                        repo=repository,
                        file_path=file_path,
                        branch=branch,
                    )
                    files.append(file_data)
                except Exception as e:
                    # Log error but continue with other files
                    files.append({
                        "path": file_path,
                        "error": str(e),
                        "content": None,
                    })

            # Store in action tags (for indexing/filtering)
            # These tag keys map to first-class fields in the backend:
            # - provider → Action.provider
            # - commit_sha → Action.externmt_id
            # - url → Action.externmt_url
            action.tags["provider"] = "github"
            action.tags["repository"] = repository
            action.tags["branch"] = branch
            if repo_info.get("commit_sha"):
                action.tags["commit_sha"] = repo_info["commit_sha"]
            if repo_info.get("url"):
                action.tags["url"] = repo_info["url"]

            # Store full context in action output metadata
            if action.output is None:
                action.output = {}

            action.output["_github_context"] = {
                "repository": repository,
                "branch": branch,
                "commit_sha": repo_info.get("commit_sha"),
                "url": repo_info.get("url"),
                "files": files,
                "fetched_at": time.time(),
            }

        finally:
            if should_close:
                await client.close()
