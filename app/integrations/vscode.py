"""VS Code integration — communicates with the NsOps VS Code extension.

The extension runs a local HTTP server (default: http://127.0.0.1:6789) inside
VS Code. This module calls that server to trigger actions in the editor.

Install the extension from vscode-extension/ in the repo root:
    cd vscode-extension && code --install-extension .

Env vars (optional — defaults work out of the box):
    VSCODE_BRIDGE_URL   default: http://127.0.0.1:6789
    VSCODE_BRIDGE_TOKEN future: bearer token for authentication (not yet used by extension)

Supported actions:
    open_file           — open a file, optionally jump to a line
    highlight_lines     — yellow-highlight lines in a file
    notify              — show a VS Code notification (info/warning/error)
    show_diff           — open a two-panel diff view
    run_in_terminal     — run a shell command in an integrated terminal
    inject_problems     — add entries to the Problems panel
    clear_highlights    — remove all NsOps decorations
    write_output        — write a message to the NsOps output channel
    status              — check whether the extension server is reachable
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_BASE_URL = os.getenv("VSCODE_BRIDGE_URL", "http://127.0.0.1:6789").rstrip("/")
_TIMEOUT  = int(os.getenv("VSCODE_BRIDGE_TIMEOUT", "5"))


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict) -> dict:
    """POST JSON payload to the VS Code extension server."""
    url  = f"{_BASE_URL}{endpoint}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return json.loads(r.read())
    except urllib.error.URLError as exc:
        return {"success": False, "error": f"VS Code extension unreachable: {exc.reason}"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _get(endpoint: str) -> dict:
    """GET request to the VS Code extension server."""
    url = f"{_BASE_URL}{endpoint}"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return json.loads(r.read())
    except urllib.error.URLError as exc:
        return {"success": False, "error": f"VS Code extension unreachable: {exc.reason}"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ── Public API ────────────────────────────────────────────────────────────────

def status() -> dict:
    """Check whether the NsOps VS Code extension is running.

    Returns:
        {
            "connected": bool,
            "workspace": str | None,
            "port": int,
            "files": int,
        }
    """
    result = _get("/status")
    if result.get("success") is False:
        return {"connected": False, "error": result.get("error")}
    return {
        "connected": result.get("status") == "running",
        "workspace": result.get("workspace"),
        "port":      result.get("port"),
        "files":     result.get("files", 0),
        "version":   result.get("version"),
    }


def open_file(
    file_path: str,
    line: Optional[int] = None,
    column: Optional[int] = None,
    preview: bool = False,
) -> dict:
    """Open a file in VS Code, optionally jumping to a specific line.

    Args:
        file_path: Absolute path to the file.
        line:      1-based line number to jump to.
        column:    1-based column number (default: start of line).
        preview:   If True, open in preview mode (closes on next open).

    Returns:
        {"success": bool, "file": str, "line": int | None}
    """
    payload: dict = {"file_path": file_path, "preview": preview}
    if line is not None:
        payload["line"] = line
    if column is not None:
        payload["column"] = column
    return _post("/open", payload)


def highlight_lines(
    file_path: str,
    lines: list[int | dict],
) -> dict:
    """Highlight lines in a file with a yellow background decoration.

    Args:
        file_path: Absolute path to the file.
        lines:     List of line numbers (1-based) OR dicts with {"line": int, "message": str}.

    Returns:
        {"success": bool, "highlighted": int}

    Example:
        highlight_lines("/app/main.py", [10, 15, 20])
        highlight_lines("/app/main.py", [{"line": 10, "message": "Crash here"}])
    """
    normalized = [
        {"line": l} if isinstance(l, int) else l
        for l in lines
    ]
    return _post("/highlight", {"file_path": file_path, "lines": normalized})


def notify(
    message: str,
    level: str = "info",
    actions: Optional[list[str]] = None,
) -> dict:
    """Show a VS Code notification.

    Args:
        message: Notification text.
        level:   "info" | "warning" | "error"
        actions: Optional list of action button labels.

    Returns:
        {"success": bool}
    """
    payload: dict = {"message": message, "level": level}
    if actions:
        payload["actions"] = actions
    return _post("/notify", payload)


def show_diff(
    title: str,
    original_path: Optional[str] = None,
    original_content: Optional[str] = None,
    modified_path: Optional[str] = None,
    modified_content: Optional[str] = None,
) -> dict:
    """Open a two-panel diff view in VS Code.

    Provide either a file path or raw content for each side.

    Args:
        title:            Title shown in the diff tab.
        original_path:    Absolute path to the original file.
        original_content: Raw text for the original side.
        modified_path:    Absolute path to the modified file.
        modified_content: Raw text for the modified side.

    Returns:
        {"success": bool, "title": str}
    """
    payload: dict = {"title": title}
    if original_path:    payload["original_path"]    = original_path
    if original_content: payload["original_content"] = original_content
    if modified_path:    payload["modified_path"]    = modified_path
    if modified_content: payload["modified_content"] = modified_content
    return _post("/diff", payload)


def run_in_terminal(
    command: str,
    name: str = "NsOps",
    cwd: Optional[str] = None,
) -> dict:
    """Run a shell command in a VS Code integrated terminal.

    Args:
        command: Shell command to execute.
        name:    Terminal panel name (reuses an existing terminal with the same name).
        cwd:     Working directory for a new terminal.

    Returns:
        {"success": bool, "terminal": str, "command": str}
    """
    payload: dict = {"command": command, "name": name}
    if cwd:
        payload["cwd"] = cwd
    return _post("/terminal", payload)


def inject_problems(
    problems: list[dict],
    source: str = "NsOps",
) -> dict:
    """Inject diagnostics into the VS Code Problems panel.

    Args:
        problems: List of dicts with keys:
                    file_path  str       absolute path
                    line       int       1-based line number
                    message    str       problem description
                    severity   str       "error" | "warning" | "info" | "hint"
        source:   Label shown in the Problems panel source column.

    Returns:
        {"success": bool, "count": int}

    Example:
        inject_problems([
            {"file_path": "/app/main.py", "line": 42,
             "message": "Unhandled exception path", "severity": "warning"}
        ])
    """
    return _post("/problems", {"source": source, "problems": problems})


def clear_highlights() -> dict:
    """Remove all NsOps line decorations from open editors.

    Returns:
        {"success": bool}
    """
    return _post("/clear-highlights", {})


def write_output(message: str, show: bool = False) -> dict:
    """Write a message to the NsOps output channel in VS Code.

    Args:
        message: Text to write.
        show:    If True, bring the output panel into focus.

    Returns:
        {"success": bool}
    """
    return _post("/output", {"message": message, "show": show})


# ── Convenience: open incident in VS Code ─────────────────────────────────────

def open_incident_context(
    incident_id: str,
    root_cause: str,
    file_path: Optional[str] = None,
    problem_line: Optional[int] = None,
) -> dict:
    """Surface an incident in VS Code — notify + optionally open the relevant file.

    This is the primary entry point called from the NsOps pipeline when an
    incident has been linked to a specific source file.

    Args:
        incident_id:  Incident ID for display.
        root_cause:   Short description of the root cause.
        file_path:    Optional source file related to the incident.
        problem_line: Optional line number in file_path.

    Returns:
        {"success": bool, "actions_taken": list[str]}
    """
    actions: list[str] = []

    # Notification
    msg = f"[{incident_id}] {root_cause}"
    nr  = notify(msg, level="warning", actions=["Open File", "Dismiss"])
    if nr.get("success"):
        actions.append("notified")

    # Open file if provided
    if file_path:
        fr = open_file(file_path, line=problem_line)
        if fr.get("success"):
            actions.append("file_opened")

        # Highlight the problem line
        if problem_line:
            hr = highlight_lines(file_path, [{"line": problem_line, "message": root_cause}])
            if hr.get("success"):
                actions.append("line_highlighted")

    # Write to output channel
    write_output(f"Incident {incident_id}: {root_cause}", show=bool(file_path))
    actions.append("output_written")

    return {"success": True, "incident_id": incident_id, "actions_taken": actions}
