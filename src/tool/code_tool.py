"""Code repository git tools."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_DEFAULT_CODE_REPO_DIR = Path("/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/code_repo")


def _normalize_repo_name(git_url: str) -> str:
    raw = str(git_url or "").strip()
    if not raw:
        return ""

    parsed = urlparse(raw)
    path = parsed.path
    if not path and ":" in raw and "@" in raw:
        # support git@host:group/repo.git
        path = raw.split(":", 1)[-1]
    name = Path(path).name if path else ""
    if name.endswith(".git"):
        name = name[:-4]
    return name.strip()


def _run_git_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    except Exception as err:  # pragma: no cover - runtime environment error
        return {
            "ok": False,
            "return_code": -1,
            "stdout": "",
            "stderr": str(err),
            "message": f"git command failed: {err}",
        }

    ok = result.returncode == 0
    return {
        "ok": ok,
        "return_code": int(result.returncode),
        "stdout": str(result.stdout or "").strip(),
        "stderr": str(result.stderr or "").strip(),
        "message": "success" if ok else "git command failed",
    }


def clone_repo(git_url: str, repo_root: str | Path = _DEFAULT_CODE_REPO_DIR) -> dict[str, Any]:
    """Clone a repository into src/code_repo by git url."""
    url = str(git_url or "").strip()
    if not url:
        return {"ok": False, "message": "empty git_url"}

    repo_name = _normalize_repo_name(url)
    if not repo_name:
        return {"ok": False, "message": f"invalid git_url: {url}"}

    root = Path(repo_root).expanduser().resolve()
    target_dir = root / repo_name
    root.mkdir(parents=True, exist_ok=True)

    if (target_dir / ".git").is_dir():
        return {
            "ok": True,
            "action": "clone",
            "status": "already_exists",
            "git_url": url,
            "target_dir": str(target_dir),
            "message": "repository already exists",
        }

    run_result = _run_git_command(["git", "clone", url, str(target_dir)], cwd=None)
    return {
        **run_result,
        "action": "clone",
        "status": "cloned" if bool(run_result.get("ok")) else "failed",
        "git_url": url,
        "target_dir": str(target_dir),
    }


def pull_repo(git_url: str, repo_root: str | Path = _DEFAULT_CODE_REPO_DIR) -> dict[str, Any]:
    """Pull latest changes for repository in src/code_repo by git url."""
    url = str(git_url or "").strip()
    if not url:
        return {"ok": False, "message": "empty git_url"}

    repo_name = _normalize_repo_name(url)
    if not repo_name:
        return {"ok": False, "message": f"invalid git_url: {url}"}

    root = Path(repo_root).expanduser().resolve()
    target_dir = root / repo_name
    git_dir = target_dir / ".git"
    if not git_dir.is_dir():
        return {
            "ok": False,
            "action": "pull",
            "status": "failed",
            "git_url": url,
            "target_dir": str(target_dir),
            "message": "repository not found, clone first",
        }

    run_result = _run_git_command(["git", "pull", "--ff-only"], cwd=target_dir)
    return {
        **run_result,
        "action": "pull",
        "status": "updated" if bool(run_result.get("ok")) else "failed",
        "git_url": url,
        "target_dir": str(target_dir),
    }

