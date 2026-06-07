"""Code repository git tools."""

from __future__ import annotations

import shutil
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


def _is_usable_local_repo(target_dir: Path) -> bool:
    git_dir = target_dir / ".git"
    if not git_dir.is_dir():
        return False
    work_tree_result = _run_git_command(["git", "rev-parse", "--is-inside-work-tree"], cwd=target_dir)
    if not bool(work_tree_result.get("ok")):
        return False
    head_result = _run_git_command(["git", "rev-parse", "--verify", "HEAD"], cwd=target_dir)
    return bool(head_result.get("ok"))


def _remove_invalid_repo_dir(target_dir: Path) -> dict[str, Any]:
    try:
        shutil.rmtree(target_dir)
    except Exception as err:  # pragma: no cover - filesystem/runtime error
        return {
            "ok": False,
            "return_code": -1,
            "stdout": "",
            "stderr": str(err),
            "message": f"failed to remove invalid repository: {err}",
        }
    return {
        "ok": True,
        "return_code": 0,
        "stdout": "",
        "stderr": "",
        "message": "invalid repository removed",
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
        if not _is_usable_local_repo(target_dir):
            remove_result = _remove_invalid_repo_dir(target_dir)
            if not bool(remove_result.get("ok")):
                return {
                    **remove_result,
                    "action": "clone",
                    "status": "failed",
                    "git_url": url,
                    "target_dir": str(target_dir),
                }
        else:
            return {
                "ok": True,
                "action": "clone",
                "status": "already_exists",
                "git_url": url,
                "target_dir": str(target_dir),
                "message": "repository already exists",
            }
    elif target_dir.exists():
        return {
            "ok": False,
            "action": "clone",
            "status": "failed",
            "git_url": url,
            "target_dir": str(target_dir),
            "message": "target directory exists but is not a git repository",
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
    if not _is_usable_local_repo(target_dir):
        return {
            "ok": False,
            "action": "pull",
            "status": "failed",
            "git_url": url,
            "target_dir": str(target_dir),
            "message": "repository invalid, reclone required",
        }

    run_result = _run_git_command(["git", "pull", "--ff-only"], cwd=target_dir)
    return {
        **run_result,
        "action": "pull",
        "status": "updated" if bool(run_result.get("ok")) else "failed",
        "git_url": url,
        "target_dir": str(target_dir),
    }


def pull_repo_local(repo_name: str, repo_root: str | Path = _DEFAULT_CODE_REPO_DIR) -> dict[str, Any]:
    """Pull latest changes for repository in src/code_repo by local repo name."""
    name = str(repo_name or "").strip()
    if not name:
        return {"ok": False, "message": "empty repo_name"}

    root = Path(repo_root).expanduser().resolve()
    target_dir = root / name
    git_dir = target_dir / ".git"
    if not git_dir.is_dir():
        return {
            "ok": False,
            "action": "pull_local",
            "status": "failed",
            "repo_name": name,
            "target_dir": str(target_dir),
            "message": "repository not found in local code_repo",
        }
    if not _is_usable_local_repo(target_dir):
        return {
            "ok": False,
            "action": "pull_local",
            "status": "failed",
            "repo_name": name,
            "target_dir": str(target_dir),
            "message": "repository invalid, reclone required",
        }

    run_result = _run_git_command(["git", "pull", "--ff-only"], cwd=target_dir)
    return {
        **run_result,
        "action": "pull_local",
        "status": "updated" if bool(run_result.get("ok")) else "failed",
        "repo_name": name,
        "target_dir": str(target_dir),
    }
