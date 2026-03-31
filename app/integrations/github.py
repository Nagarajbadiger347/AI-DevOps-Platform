"""GitHub integration — profile-level + per-repo access.

GITHUB_REPO can be:
  - A profile URL:  https://github.com/Nagarajbadiger347   → works across all repos
  - A repo URL:     https://github.com/Nagarajbadiger347/my-repo → scoped to one repo
  - A repo slug:    Nagarajbadiger347/my-repo               → scoped to one repo
  - Empty:          → profile-level using token owner
"""

import base64
import datetime
import os

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from github import Github
from github.GithubException import GithubException

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_REPO  = os.getenv("GITHUB_REPO", "").strip()


# ── URL / slug parsing ────────────────────────────────────────

def _parse_github_url(raw: str):
    """Return (owner, repo_name_or_None).

    Profile URL  → (owner, None)
    Repo URL     → (owner, repo)
    owner/repo   → (owner, repo)
    username     → (username, None)
    """
    raw = raw.strip().rstrip("/").removesuffix(".git")

    if raw.startswith("http"):
        parts = [p for p in raw.split("/") if p and p not in ("https:", "http:", "github.com")]
        # parts[0] = owner, parts[1] (if present) = repo
        if len(parts) == 0:
            return (None, None)
        owner = parts[0]
        repo  = parts[1] if len(parts) >= 2 else None
        return (owner, repo)

    if "/" in raw:
        bits = raw.split("/", 1)
        return (bits[0], bits[1])

    # bare username
    return (raw or None, None)


def _client() -> Github:
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN not configured — add it to .env")
    return Github(GITHUB_TOKEN)


def _owner_and_repo():
    """Return (owner, optional_repo_object).

    If GITHUB_REPO points to a profile, repo is None and callers
    iterate across all user repos themselves.
    """
    owner, repo_name = _parse_github_url(GITHUB_REPO) if GITHUB_REPO else (None, None)
    gh = _client()

    if not owner:
        # Fall back to the authenticated user
        owner = gh.get_user().login

    if repo_name:
        return owner, gh.get_repo(f"{owner}/{repo_name}")
    return owner, None


def _all_repos(owner: str, limit: int = 30):
    """Yield up to `limit` repos for the owner, sorted by latest push."""
    gh = _client()
    try:
        user  = gh.get_user(owner)
        repos = sorted(user.get_repos(type="owner"), key=lambda r: r.pushed_at or datetime.datetime.min, reverse=True)
        return list(repos[:limit])
    except GithubException as e:
        raise RuntimeError(str(e)) from e


def _pick_repo(repo_name: str = ""):
    """Return a single repo object.

    Priority:
      1. `repo_name` argument (passed explicitly by caller)
      2. GITHUB_REPO if it contains a specific repo
      3. Most-recently-pushed repo in the account
    """
    owner, default_repo = _owner_and_repo()
    if repo_name:
        return _client().get_repo(f"{owner}/{repo_name}")
    if default_repo:
        return default_repo
    # Profile-level: use most recently pushed repo
    repos = _all_repos(owner, limit=1)
    if not repos:
        raise RuntimeError(f"No repositories found for {owner}")
    return repos[0]


# ── Observability ─────────────────────────────────────────────

def list_repos() -> dict:
    """List all repositories for the configured GitHub account."""
    try:
        owner, _ = _owner_and_repo()
        repos = _all_repos(owner, limit=50)
        result = [
            {
                "name":        r.name,
                "full_name":   r.full_name,
                "description": r.description or "",
                "url":         r.html_url,
                "language":    r.language or "",
                "stars":       r.stargazers_count,
                "forks":       r.forks_count,
                "open_issues": r.open_issues_count,
                "pushed_at":   r.pushed_at.isoformat() if r.pushed_at else None,
                "private":     r.private,
            }
            for r in repos
        ]
        return {"success": True, "owner": owner, "repos": result, "count": len(result)}
    except RuntimeError as e:
        return {"success": False, "error": str(e)}
    except GithubException as e:
        return {"success": False, "error": str(e)}


def get_recent_commits(hours: int = 2, branch: str = "", repo_name: str = "") -> dict:
    """Return commits pushed in the last N hours across all repos (or a specific one)."""
    try:
        owner, default_repo = _owner_and_repo()
        since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)

        # If a specific repo is configured / requested, scan only that one
        if repo_name or default_repo:
            target_repos = [_pick_repo(repo_name)]
        else:
            # Profile-level: scan top 10 most recently pushed repos
            target_repos = _all_repos(owner, limit=10)

        all_commits = []
        for repo in target_repos:
            try:
                kwargs = {"since": since}
                if branch:
                    kwargs["sha"] = branch
                for c in repo.get_commits(**kwargs):
                    all_commits.append({
                        "repo":          repo.full_name,
                        "sha":           c.sha[:10],
                        "message":       c.commit.message.split("\n")[0],
                        "author":        c.commit.author.name,
                        "date":          c.commit.author.date.isoformat(),
                        "url":           c.html_url,
                        "files_changed": [f.filename for f in c.files],
                    })
            except GithubException:
                continue

        all_commits.sort(key=lambda x: x["date"], reverse=True)
        return {"success": True, "commits": all_commits, "count": len(all_commits),
                "hours": hours, "owner": owner}
    except RuntimeError as e:
        return {"success": False, "error": str(e)}
    except GithubException as e:
        return {"success": False, "error": str(e)}


def get_recent_prs(hours: int = 24, state: str = "closed", repo_name: str = "") -> dict:
    """Return recently merged PRs across all repos (or a specific one)."""
    try:
        owner, default_repo = _owner_and_repo()
        since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)

        if repo_name or default_repo:
            target_repos = [_pick_repo(repo_name)]
        else:
            target_repos = _all_repos(owner, limit=10)

        all_prs = []
        for repo in target_repos:
            try:
                for pr in repo.get_pulls(state=state, sort="updated", direction="desc"):
                    if pr.merged_at and pr.merged_at > since:
                        all_prs.append({
                            "repo":      repo.full_name,
                            "number":    pr.number,
                            "title":     pr.title,
                            "author":    pr.user.login,
                            "merged_at": pr.merged_at.isoformat(),
                            "url":       pr.html_url,
                            "files":     [f.filename for f in pr.get_files()],
                            "additions": pr.additions,
                            "deletions": pr.deletions,
                        })
            except GithubException:
                continue

        all_prs.sort(key=lambda x: x["merged_at"], reverse=True)
        return {"success": True, "prs": all_prs, "count": len(all_prs),
                "hours": hours, "owner": owner}
    except RuntimeError as e:
        return {"success": False, "error": str(e)}
    except GithubException as e:
        return {"success": False, "error": str(e)}


def get_commit_diff(sha: str, repo_name: str = "") -> dict:
    """Get the full diff for a commit."""
    try:
        repo   = _pick_repo(repo_name)
        commit = repo.get_commit(sha)
        files  = [
            {
                "filename":  f.filename,
                "status":    f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "patch":     f.patch or "",
            }
            for f in commit.files
        ]
        return {
            "success": True,
            "repo":    repo.full_name,
            "sha":     sha,
            "message": commit.commit.message,
            "author":  commit.commit.author.name,
            "date":    commit.commit.author.date.isoformat(),
            "files":   files,
        }
    except RuntimeError as e:
        return {"success": False, "error": str(e)}
    except GithubException as e:
        return {"success": False, "error": str(e)}


def get_file_content(file_path: str, ref: str = "", repo_name: str = "") -> dict:
    """Get content of a file at a given branch/commit ref."""
    try:
        repo    = _pick_repo(repo_name)
        kwargs  = {"ref": ref} if ref else {}
        content = repo.get_contents(file_path, **kwargs)
        decoded = base64.b64decode(content.content).decode("utf-8", errors="replace")
        return {"success": True, "repo": repo.full_name,
                "path": file_path, "content": decoded, "sha": content.sha}
    except RuntimeError as e:
        return {"success": False, "error": str(e)}
    except GithubException as e:
        return {"success": False, "error": str(e)}


def get_profile_summary() -> dict:
    """Return a summary of the GitHub account — repos, activity, languages."""
    try:
        owner, _ = _owner_and_repo()
        gh   = _client()
        user = gh.get_user(owner)
        repos = _all_repos(owner, limit=50)

        languages: dict = {}
        total_stars = 0
        total_issues = 0
        for r in repos:
            total_stars  += r.stargazers_count
            total_issues += r.open_issues_count
            if r.language:
                languages[r.language] = languages.get(r.language, 0) + 1

        top_langs = sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "success":       True,
            "username":      user.login,
            "name":          user.name or "",
            "bio":           user.bio or "",
            "public_repos":  user.public_repos,
            "followers":     user.followers,
            "following":     user.following,
            "total_stars":   total_stars,
            "open_issues":   total_issues,
            "top_languages": dict(top_langs),
            "profile_url":   user.html_url,
        }
    except RuntimeError as e:
        return {"success": False, "error": str(e)}
    except GithubException as e:
        return {"success": False, "error": str(e)}


# ── Actions ───────────────────────────────────────────────────

def create_issue(title: str = "AI DevOps Issue",
                 body: str = "Generated by AI Orchestrator",
                 repo_name: str = "") -> dict:
    try:
        repo  = _pick_repo(repo_name)
        issue = repo.create_issue(title=title, body=body)
        return {"success": True, "issue_number": issue.number,
                "url": issue.html_url, "repo": repo.full_name}
    except RuntimeError as e:
        return {"success": False, "error": str(e)}
    except GithubException as e:
        return {"success": False, "error": str(e)}


def create_pull_request(head: str, base: str = "main",
                        title: str = "AI DevOps PR",
                        body: str = "Generated by AI Orchestrator",
                        repo_name: str = "") -> dict:
    try:
        repo = _pick_repo(repo_name)
        pr   = repo.create_pull(title=title, body=body, head=head, base=base)
        return {"success": True, "pr_number": pr.number,
                "url": pr.html_url, "repo": repo.full_name}
    except RuntimeError as e:
        return {"success": False, "error": str(e)}
    except GithubException as e:
        return {"success": False, "error": str(e)}


def create_incident_pr(incident_id: str, title: str, body: str,
                       base: str = "main", file_changes: list = None,
                       repo_name: str = "") -> dict:
    """Create a branch + PR for an incident fix or report."""
    try:
        repo        = _pick_repo(repo_name)
        branch_name = f"incident/{incident_id.lower().replace(' ', '-')}"
        base_sha    = repo.get_branch(base).commit.sha

        try:
            repo.get_branch(branch_name)
        except GithubException:
            repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_sha)

        changes = file_changes or [{
            "path":    f"incidents/{incident_id}/report.md",
            "content": f"# Incident Report: {incident_id}\n\n{body}",
        }]

        for fc in changes:
            path, content = fc["path"], fc["content"]
            try:
                existing = repo.get_contents(path, ref=branch_name)
                repo.update_file(path, f"incident({incident_id}): update {path}",
                                 content, existing.sha, branch=branch_name)
            except GithubException:
                repo.create_file(path, f"incident({incident_id}): add {path}",
                                 content, branch=branch_name)

        pr = repo.create_pull(title=title, body=body, head=branch_name, base=base)
        return {
            "success":   True,
            "branch":    branch_name,
            "pr_number": pr.number,
            "url":       pr.html_url,
            "repo":      repo.full_name,
        }
    except RuntimeError as e:
        return {"success": False, "error": str(e)}
    except GithubException as e:
        return {"success": False, "error": str(e)}


def get_pr_for_review(pr_number: int, repo_name: str = "") -> dict:
    """Fetch a PR's metadata, changed files, and diffs for AI review."""
    try:
        repo  = _pick_repo(repo_name)
        pr    = repo.get_pull(pr_number)
        files = [
            {
                "filename":  f.filename,
                "status":    f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "patch":     (f.patch or "")[:3000],
            }
            for f in pr.get_files()
        ]
        return {
            "success":     True,
            "repo":        repo.full_name,
            "number":      pr.number,
            "title":       pr.title,
            "author":      pr.user.login,
            "base_branch": pr.base.ref,
            "head_branch": pr.head.ref,
            "body":        pr.body or "",
            "additions":   pr.additions,
            "deletions":   pr.deletions,
            "url":         pr.html_url,
            "files":       files,
        }
    except RuntimeError as e:
        return {"success": False, "error": str(e)}
    except GithubException as e:
        return {"success": False, "error": str(e)}


def post_pr_review_comment(pr_number: int, body: str, repo_name: str = "") -> dict:
    """Post an AI-generated review comment on a PR."""
    try:
        repo = _pick_repo(repo_name)
        pr   = repo.get_pull(pr_number)
        pr.create_issue_comment(body)
        return {"success": True, "pr_number": pr_number,
                "url": pr.html_url, "repo": repo.full_name}
    except RuntimeError as e:
        return {"success": False, "error": str(e)}
    except GithubException as e:
        return {"success": False, "error": str(e)}
