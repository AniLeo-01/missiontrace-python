"""GitHub MCP client for fetching repository context."""
from __future__ import annotations

import os
import time
from typing import Any
from dataclasses import dataclass

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


@dataclass
class GitHubContext:
    """Represents GitHub repository context for an action."""
    repository: str  # "owner/repo"
    branch: str
    commit_sha: str | None = None
    files: list[dict[str, Any]] | None = None  # [{path, content, start_line, end_line}]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "files": self.files or [],
        }


class GitHubMCPClient:
    """Client for fetching code context from GitHub via MCP server.

    Requires a GitHub MCP server to be running with OAuth authentication.

    Authentication:
        1. Authenticate via frontend: http://localhost:3000/settings
        2. After OAuth, the session token is stored
        3. Set GITHUB_MCP_SESSION environment variable with the session token
        4. SDK will use this token to authenticate with MCP server

    Usage:
        # After authenticating via frontend
        export GITHUB_MCP_SESSION=your_session_token

        client = GitHubMCPClient(mcp_endpoint="http://localhost:3001")
        context = await client.fetch_file_content(
            repo="anthropics/anthropic-sdk-python",
            file_path="src/anthropic/client.py",
            start_line=10,
            end_line=50
        )
    """

    def __init__(
        self,
        mcp_endpoint: str | None = None,
        session_token: str | None = None,
    ):
        if not HAS_HTTPX:
            raise ImportError("httpx is required for GitHub MCP integration. Install with: pip install httpx")

        self.mcp_endpoint = mcp_endpoint or os.environ.get("GITHUB_MCP_ENDPOINT", "http://localhost:3001")
        self.session_token = session_token or os.environ.get("GITHUB_MCP_SESSION")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def fetch_file_content(
        self,
        repo: str,  # "owner/repo"
        file_path: str,
        branch: str = "main",
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Fetch file content from GitHub via MCP.

        Returns:
            {
                "path": "src/app.py",
                "content": "def foo():\\n  pass",
                "start_line": 10,
                "end_line": 20,
                "totmt_lines": 500,
                "url": "https://github.com/owner/repo/blob/main/src/app.py#L10-L20"
            }
        """
        # Call GitHub MCP server
        # Note: Actual MCP protocol may differ - this is a placeholder for the pattern
        response = await self._client.post(
            f"{self.mcp_endpoint}/github/read_file",
            json={
                "repository": repo,
                "path": file_path,
                "branch": branch,
                "start_line": start_line,
                "end_line": end_line,
            },
            headers={"Authorization": f"Bearer {self.session_token}"} if self.session_token else {},
        )
        response.raise_for_status()
        return response.json()

    async def fetch_repository_info(self, repo: str, branch: str = "main") -> dict[str, Any]:
        """Fetch repository metadata.

        Returns:
            {
                "repository": "owner/repo",
                "branch": "main",
                "commit_sha": "abc123def456...",
                "url": "https://github.com/owner/repo"
            }
        """
        response = await self._client.post(
            f"{self.mcp_endpoint}/github/repo_info",
            json={"repository": repo, "branch": branch},
            headers={"Authorization": f"Bearer {self.session_token}"} if self.session_token else {},
        )
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self._client.aclose()

    def __del__(self):
        # Attempt to close (may not work in all cases, but helpful)
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._client.aclose())
        except Exception:
            pass
