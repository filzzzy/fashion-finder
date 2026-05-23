from __future__ import annotations

from pathlib import Path


def get_git_commit_id(repo_root: str | Path | None = None) -> str:
    try:
        import git
    except ImportError:
        return "unknown"
    try:
        repo = git.Repo(
            str(repo_root) if repo_root else Path.cwd(), search_parent_directories=True
        )
        return repo.head.commit.hexsha
    except (git.InvalidGitRepositoryError, git.NoSuchPathError, ValueError):
        return "unknown"
