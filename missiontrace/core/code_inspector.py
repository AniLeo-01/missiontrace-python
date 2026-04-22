"""
Code inspection utilities for capturing function metadata.

This module provides utilities to inspect the calling code and extract
metadata like docstrings, file paths, source code, etc. This metadata
is used to build rich Action Context using LLMs.
"""

from __future__ import annotations

import inspect
import os
from typing import Any
from dataclasses import dataclass


@dataclass
class FunctionMetadata:
    """Metadata about a function captured during action execution."""

    function_name: str | None = None
    module_name: str | None = None
    file_path: str | None = None
    line_number: int | None = None
    docstring: str | None = None
    source_code: str | None = None
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "function_name": self.function_name,
            "module_name": self.module_name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "docstring": self.docstring,
            "source_code": self.source_code,
            "signature": self.signature,
        }


def capture_function_metadata(
    skip_frames: int = 2,
    capture_source: bool = False,
    max_source_lines: int = 100,
) -> FunctionMetadata:
    """
    Capture metadata about the calling function.

    Args:
        skip_frames: Number of frames to skip (default 2: this function + action context)
        capture_source: Whether to capture the full source code
        max_source_lines: Maximum lines of source code to capture

    Returns:
        FunctionMetadata object with captured information

    Example:
        with missiontrace.action("tool_call", name="process") as act:
            metadata = capture_function_metadata()
            # metadata.docstring contains the docstring of the calling function
    """
    metadata = FunctionMetadata()

    try:
        # Get the call stack
        frame = inspect.currentframe()
        if frame is None:
            return metadata

        # Skip frames to get to the actual calling code
        for _ in range(skip_frames):
            if frame.f_back is None:
                return metadata
            frame = frame.f_back

        # Extract frame information
        frame_info = inspect.getframeinfo(frame)
        metadata.file_path = frame_info.filename
        metadata.line_number = frame_info.lineno

        # Get the function object if possible
        func = None

        # Try to get the function from local variables
        if "self" in frame.f_locals:
            # Method call
            obj = frame.f_locals["self"]
            func_name = frame.f_code.co_name
            if hasattr(obj, func_name):
                func = getattr(obj, func_name)
                metadata.function_name = f"{obj.__class__.__name__}.{func_name}"
                metadata.module_name = obj.__class__.__module__
        else:
            # Function call
            func_name = frame.f_code.co_name
            metadata.function_name = func_name

            # Try to find the function in globals
            if func_name in frame.f_globals:
                func = frame.f_globals[func_name]
                if hasattr(func, "__module__"):
                    metadata.module_name = func.__module__

        # Extract docstring
        if func is not None:
            try:
                doc = inspect.getdoc(func)
                if doc:
                    metadata.docstring = doc
            except Exception:
                pass

            # Extract signature
            try:
                sig = inspect.signature(func)
                metadata.signature = f"{metadata.function_name}{sig}"
            except Exception:
                pass

            # Extract source code if requested
            if capture_source:
                try:
                    source_lines = inspect.getsourcelines(func)[0]
                    # Limit source code length
                    if len(source_lines) > max_source_lines:
                        source_lines = source_lines[:max_source_lines]
                        source_lines.append("... (truncated)")
                    metadata.source_code = "".join(source_lines)
                except Exception:
                    pass

    except Exception:
        # If anything fails, return empty metadata
        pass

    return metadata


def get_repository_info(file_path: str | None) -> dict[str, Any] | None:
    """
    Extract repository info from the file's directory structure.

    Uses the git root directory name as the repository name (simpler and works
    for most cases). The backend can use GitHub MCP to fetch additional context
    if needed.

    Args:
        file_path: Path to the source file

    Returns:
        Dictionary with repository information or None
    """
    if not file_path or not os.path.exists(file_path):
        return None

    try:
        import subprocess

        # Get git repository root
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=os.path.dirname(file_path),
            capture_output=True,
            text=True,
            timeout=2,
        )

        if result.returncode != 0:
            return None

        repo_root = result.stdout.strip()

        # Repository name = directory name (simple and reliable)
        repo_name = os.path.basename(repo_root)

        # Get current commit SHA
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=2,
        )
        commit_sha = result.stdout.strip() if result.returncode == 0 else None

        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=2,
        )
        branch = result.stdout.strip() if result.returncode == 0 else None

        # Get relative path within repository
        rel_path = os.path.relpath(file_path, repo_root)

        # Try to get GitHub remote URL
        github_url = None
        github_repository = None
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            remote_url = result.stdout.strip()
            # Parse GitHub URL from remote (handles both HTTPS and SSH formats)
            # HTTPS: https://github.com/owner/repo.git
            # SSH: git@github.com:owner/repo.git
            import re
            match = re.match(r"(?:https://github\.com/|git@github\.com:)([^/]+)/([^/]+?)(?:\.git)?$", remote_url)
            if match:
                owner, repo = match.groups()
                github_repository = f"{owner}/{repo}"
                github_url = f"https://github.com/{owner}/{repo}"

        return {
            "repo_name": repo_name,
            "repo_root": repo_root,
            "commit_sha": commit_sha,
            "branch": branch,
            "file_path_relative": rel_path,
            "github_repository": github_repository,
            "github_url": github_url,
        }

    except Exception:
        return None
