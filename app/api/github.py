"""
GitHub integration routes.
Paths: /github/*
"""
import os

from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import require_viewer, AuthContext

router = APIRouter(tags=["github"])


@router.get("/github/status")
def github_status(auth: AuthContext = Depends(require_viewer)):
    """Configuration status of the GitHub integration. Used by the dashboard
    onboarding flow to decide whether GitHub is already connected."""
    token = os.getenv("GITHUB_TOKEN", "").strip()
    repo  = os.getenv("GITHUB_REPO", "").strip()
    slug  = None
    error = None
    if repo:
        try:
            from app.integrations.github import _parse_github_url
            owner, repo_name = _parse_github_url(repo)
            slug = f"{owner}/{repo_name}" if repo_name else f"{owner} (profile-level)"
        except Exception as exc:
            error = str(exc)
    configured = bool(token and slug)
    return {
        "configured": configured,
        "token_set":  bool(token),
        "repo_raw":   repo or None,
        "repo_parsed": slug,
        "error":      error,
    }


@router.get("/github/repos")
def github_repos(auth: AuthContext = Depends(require_viewer)):
    """List all repositories for the configured GitHub account."""
    from app.integrations.github import list_repos
    return list_repos()


@router.get("/github/profile")
def github_profile(auth: AuthContext = Depends(require_viewer)):
    """GitHub account summary — repos, stars, top languages."""
    from app.integrations.github import get_profile_summary
    return get_profile_summary()


@router.get("/github/commits")
def github_commits(hours: int = 24, repo: str = "", auth: AuthContext = Depends(require_viewer)):
    """Recent commits across all repos (or a specific one)."""
    from app.integrations.github import get_recent_commits
    return get_recent_commits(hours=hours, repo_name=repo)


@router.get("/github/prs")
def github_prs(
    hours: int = 48,
    state: str = "closed",
    repo: str = "",
    auth: AuthContext = Depends(require_viewer),
):
    """Recent PRs across all repos (or a specific one). `state` is passed
    through to PyGithub: open | closed | all (the underlying SDK rejects
    'merged' so we collapse it to 'closed' and rely on `merged_at` filtering)."""
    from app.integrations.github import get_recent_prs
    if state not in ("open", "closed", "all"):
        state = "closed"
    return get_recent_prs(hours=hours, state=state, repo_name=repo)


@router.get("/github/pr/{pr_number}/review")
def github_pr_review(pr_number: int, auth: AuthContext = Depends(require_viewer)):
    """Get AI review for a PR by number."""
    from app.integrations.github import get_pr_for_review
    from app.llm.claude import review_pr
    data = get_pr_for_review(pr_number)
    if not data.get("success"):
        raise HTTPException(status_code=404, detail=data.get("error", "PR not found"))
    review = review_pr(data)
    return {"pr": pr_number, "review": review, "pr_data": data}


@router.post("/github/issue")
def github_issue(title: str = "AI DevOps Issue", body: str = "", repo: str = "",
                 auth: AuthContext = Depends(require_viewer)):
    from app.integrations.github import create_issue
    return create_issue(title=title, body=body, repo_name=repo)
