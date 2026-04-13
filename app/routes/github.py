"""
GitHub integration routes.
Paths: /github/*
"""
from fastapi import APIRouter, Depends, HTTPException
from app.routes.deps import require_viewer, AuthContext

router = APIRouter(tags=["github"])


@router.get("/github/repos")
def github_repos(auth: AuthContext = Depends(require_viewer)):
    """List all repositories for the configured GitHub account."""
    from app.integrations.github import list_repos
    return list_repos()


@router.get("/github/profile")
def github_profile(auth: AuthContext = Depends(require_viewer)):
    """GitHub account summary — repos, stars, top languages."""
    from app.integrations.github import get_github_profile
    return get_github_profile()


@router.get("/github/commits")
def github_commits(hours: int = 24, repo: str = "", auth: AuthContext = Depends(require_viewer)):
    """Recent commits across all repos (or a specific one)."""
    from app.integrations.github import get_recent_commits
    return get_recent_commits(hours=hours, repo_name=repo)


@router.get("/github/prs")
def github_prs(hours: int = 48, repo: str = "", auth: AuthContext = Depends(require_viewer)):
    """Recent merged PRs across all repos (or a specific one)."""
    from app.integrations.github import get_recent_prs
    return get_recent_prs(hours=hours, repo_name=repo)


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
