"""
VS Code integration routes.
Paths: /vscode/*
"""
from fastapi import APIRouter, Depends, Request
from app.api.deps import require_viewer, require_developer, AuthContext

router = APIRouter(prefix="/vscode", tags=["vscode"])


def _vscode():
    from app.integrations import vscode as _vs
    return _vs


@router.get("/status")
async def vscode_status(u=Depends(require_viewer)):
    vs = _vscode()
    return vs.status()


@router.post("/notify")
async def vscode_notify(req: Request, u=Depends(require_viewer)):
    body = await req.json()
    vs = _vscode()
    return vs.notify(
        message=body.get("message", ""),
        level=body.get("level", "info"),
        actions=body.get("actions"),
    )


@router.post("/open")
async def vscode_open(req: Request, u=Depends(require_developer)):
    body = await req.json()
    vs = _vscode()
    return vs.open_file(
        file_path=body.get("file_path", ""),
        line=body.get("line"),
        column=body.get("column"),
        preview=body.get("preview", False),
    )


@router.post("/highlight")
async def vscode_highlight(req: Request, u=Depends(require_developer)):
    body = await req.json()
    vs = _vscode()
    return vs.highlight_lines(
        file_path=body.get("file_path", ""),
        lines=body.get("lines", []),
    )


@router.post("/terminal")
async def vscode_terminal(req: Request, u=Depends(require_developer)):
    body = await req.json()
    vs = _vscode()
    return vs.run_in_terminal(
        command=body.get("command", ""),
        name=body.get("name", "NsOps"),
        cwd=body.get("cwd"),
    )


@router.post("/diff")
async def vscode_diff(req: Request, u=Depends(require_developer)):
    body = await req.json()
    vs = _vscode()
    return vs.show_diff(
        title=body.get("title", "NsOps Diff"),
        original_path=body.get("original_path"),
        original_content=body.get("original_content"),
        modified_path=body.get("modified_path"),
        modified_content=body.get("modified_content"),
    )


@router.post("/problems")
async def vscode_problems(req: Request, u=Depends(require_developer)):
    body = await req.json()
    vs = _vscode()
    return vs.inject_problems(
        problems=body.get("problems", []),
        source=body.get("source", "NsOps"),
    )


@router.post("/clear-highlights")
async def vscode_clear_highlights(u=Depends(require_developer)):
    vs = _vscode()
    return vs.clear_highlights()


@router.post("/output")
async def vscode_output(req: Request, u=Depends(require_viewer)):
    body = await req.json()
    vs = _vscode()
    return vs.write_output(
        message=body.get("message", ""),
        show=body.get("show", False),
    )


@router.post("/incident/{incident_id}")
async def vscode_open_incident(incident_id: str, req: Request, u=Depends(require_developer)):
    body = await req.json()
    vs = _vscode()
    return vs.open_incident_context(
        incident_id=incident_id,
        root_cause=body.get("root_cause", ""),
        file_path=body.get("file_path"),
        problem_line=body.get("problem_line"),
    )
