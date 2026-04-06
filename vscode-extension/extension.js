// NsOps VS Code Extension
// Starts a local HTTP server that the NsOps platform calls to trigger VS Code actions.
//
// Supported endpoints:
//   GET  /status              — health check, returns version + workspace info
//   POST /open                — open a file (optionally at a specific line)
//   POST /highlight           — highlight / decorate lines in a file
//   POST /notify              — show a VS Code notification
//   POST /diff                — show an inline diff view
//   POST /terminal            — run a command in an integrated terminal
//   POST /problems            — inject diagnostics into the Problems panel
//   POST /clear-highlights    — remove all NsOps decorations
//   POST /output              — write a message to the NsOps output channel

const vscode = require('vscode');
const http   = require('http');

// ── State ─────────────────────────────────────────────────────────────────────
let server       = null;
let statusBar    = null;
let outputCh     = null;
let decorType    = null;  // reused decoration type for highlights

// ── Activation ────────────────────────────────────────────────────────────────
function activate(context) {
  outputCh  = vscode.window.createOutputChannel('NsOps');
  decorType = vscode.window.createTextEditorDecorationType({
    backgroundColor: 'rgba(255, 200, 0, 0.18)',
    borderWidth:     '1px',
    borderStyle:     'solid',
    borderColor:     'rgba(255, 200, 0, 0.5)',
    overviewRulerColor: 'rgba(255, 200, 0, 0.8)',
    overviewRulerLane:  vscode.OverviewRulerLane.Right,
  });

  context.subscriptions.push(
    vscode.commands.registerCommand('nsops.startServer', () => startServer(context)),
    vscode.commands.registerCommand('nsops.stopServer',  stopServer),
    vscode.commands.registerCommand('nsops.showStatus',  showStatus),
  );

  // Status bar item
  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBar.command = 'nsops.showStatus';
  context.subscriptions.push(statusBar);
  setStatusBar('stopped');

  const cfg = vscode.workspace.getConfiguration('nsops');
  if (cfg.get('autoStart', true)) {
    startServer(context);
  }
}

function deactivate() {
  stopServer();
}

// ── Server lifecycle ──────────────────────────────────────────────────────────
function startServer(context) {
  if (server) { vscode.window.showInformationMessage('NsOps server already running.'); return; }

  const port = vscode.workspace.getConfiguration('nsops').get('serverPort', 6789);

  server = http.createServer((req, res) => {
    // CORS headers so the platform dashboard can call from browser
    res.setHeader('Access-Control-Allow-Origin',  '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end',  () => {
      let payload = {};
      try { payload = body ? JSON.parse(body) : {}; } catch (_) {}
      route(req.method, req.url, payload, res);
    });
  });

  server.on('error', err => {
    vscode.window.showErrorMessage(`NsOps server error: ${err.message}`);
    setStatusBar('error');
    server = null;
  });

  server.listen(port, '127.0.0.1', () => {
    log(`NsOps integration server started on http://127.0.0.1:${port}`);
    setStatusBar('running', port);
  });
}

function stopServer() {
  if (!server) return;
  server.close(() => { log('NsOps server stopped.'); setStatusBar('stopped'); });
  server = null;
}

// ── Router ────────────────────────────────────────────────────────────────────
function route(method, url, body, res) {
  const path = url.split('?')[0];

  if (method === 'GET' && path === '/status') return handleStatus(res);

  if (method === 'POST') {
    switch (path) {
      case '/open':             return handleOpen(body, res);
      case '/highlight':        return handleHighlight(body, res);
      case '/notify':           return handleNotify(body, res);
      case '/diff':             return handleDiff(body, res);
      case '/terminal':         return handleTerminal(body, res);
      case '/problems':         return handleProblems(body, res);
      case '/clear-highlights': return handleClearHighlights(res);
      case '/output':           return handleOutput(body, res);
      default:                  return json(res, 404, { error: `Unknown endpoint: ${path}` });
    }
  }
  json(res, 404, { error: 'Not found' });
}

// ── Handlers ──────────────────────────────────────────────────────────────────

function handleStatus(res) {
  const ws    = vscode.workspace.workspaceFolders;
  const port  = vscode.workspace.getConfiguration('nsops').get('serverPort', 6789);
  json(res, 200, {
    status:    'running',
    version:   '1.0.0',
    port,
    workspace: ws ? ws[0].uri.fsPath : null,
    files:     vscode.workspace.textDocuments.length,
  });
}

async function handleOpen(body, res) {
  // body: { file_path, line?, column?, preview? }
  const filePath = body.file_path;
  if (!filePath) return json(res, 400, { error: 'file_path required' });

  try {
    const uri  = vscode.Uri.file(filePath);
    const doc  = await vscode.workspace.openTextDocument(uri);
    const opts = {};
    if (body.preview !== undefined) opts.preview = !!body.preview;

    const editor = await vscode.window.showTextDocument(doc, opts);

    if (body.line !== undefined) {
      const line   = Math.max(0, parseInt(body.line) - 1);  // 1-based → 0-based
      const col    = body.column ? Math.max(0, parseInt(body.column) - 1) : 0;
      const pos    = new vscode.Position(line, col);
      editor.selection = new vscode.Selection(pos, pos);
      editor.revealRange(new vscode.Range(pos, pos), vscode.TextEditorRevealType.InCenter);
    }

    log(`Opened: ${filePath}${body.line ? `:${body.line}` : ''}`);
    json(res, 200, { success: true, file: filePath, line: body.line || null });
  } catch (err) {
    json(res, 500, { error: err.message });
  }
}

async function handleHighlight(body, res) {
  // body: { file_path, lines: [{ line, message? }], color? }
  const filePath = body.file_path;
  const lines    = body.lines || [];
  if (!filePath || !lines.length) return json(res, 400, { error: 'file_path and lines required' });

  try {
    const uri    = vscode.Uri.file(filePath);
    const doc    = await vscode.workspace.openTextDocument(uri);
    const editor = await vscode.window.showTextDocument(doc);

    const ranges = lines.map(l => {
      const lineNum = Math.max(0, parseInt(l.line || l) - 1);
      const lineText = doc.lineAt(Math.min(lineNum, doc.lineCount - 1));
      return new vscode.Range(lineNum, 0, lineNum, lineText.text.length);
    });

    editor.setDecorations(decorType, ranges);

    // Show hover messages if provided
    const withMsg = lines.filter(l => l.message);
    if (withMsg.length) {
      const msgs = withMsg.map(l => `Line ${l.line}: ${l.message}`).join('\n');
      vscode.window.showInformationMessage(`NsOps highlighted ${ranges.length} line(s):\n${msgs}`);
    }

    log(`Highlighted ${ranges.length} lines in ${filePath}`);
    json(res, 200, { success: true, highlighted: ranges.length });
  } catch (err) {
    json(res, 500, { error: err.message });
  }
}

function handleNotify(body, res) {
  // body: { message, level?, actions? }
  const msg     = body.message || 'NsOps notification';
  const level   = (body.level || 'info').toLowerCase();
  const actions = body.actions || [];

  let fn;
  if (level === 'error')   fn = vscode.window.showErrorMessage;
  else if (level === 'warning') fn = vscode.window.showWarningMessage;
  else fn = vscode.window.showInformationMessage;

  if (actions.length) {
    fn(msg, ...actions).then(selected => {
      log(`Notification action selected: ${selected || 'dismissed'}`);
    });
  } else {
    fn(msg);
  }

  log(`Notification [${level}]: ${msg}`);
  json(res, 200, { success: true });
}

async function handleDiff(body, res) {
  // body: { title, original_path?, original_content?, modified_path?, modified_content? }
  const title = body.title || 'NsOps Diff';

  try {
    let leftUri, rightUri;

    if (body.original_path) {
      leftUri = vscode.Uri.file(body.original_path);
    } else if (body.original_content !== undefined) {
      leftUri = vscode.Uri.parse(`untitled:original-${Date.now()}`);
      // Write to a temp file via workspace edit
      const edit = new vscode.WorkspaceEdit();
      edit.createFile(leftUri, { ignoreIfExists: true });
      edit.insert(leftUri, new vscode.Position(0, 0), body.original_content);
      await vscode.workspace.applyEdit(edit);
    }

    if (body.modified_path) {
      rightUri = vscode.Uri.file(body.modified_path);
    } else if (body.modified_content !== undefined) {
      rightUri = vscode.Uri.parse(`untitled:modified-${Date.now()}`);
      const edit = new vscode.WorkspaceEdit();
      edit.createFile(rightUri, { ignoreIfExists: true });
      edit.insert(rightUri, new vscode.Position(0, 0), body.modified_content);
      await vscode.workspace.applyEdit(edit);
    }

    if (!leftUri || !rightUri) return json(res, 400, { error: 'Provide original and modified (path or content)' });

    await vscode.commands.executeCommand('vscode.diff', leftUri, rightUri, title);
    log(`Opened diff: ${title}`);
    json(res, 200, { success: true, title });
  } catch (err) {
    json(res, 500, { error: err.message });
  }
}

function handleTerminal(body, res) {
  // body: { command, name?, cwd? }
  const command = body.command;
  if (!command) return json(res, 400, { error: 'command required' });

  const name = body.name || 'NsOps';
  let term = vscode.window.terminals.find(t => t.name === name);
  if (!term) {
    const opts = { name };
    if (body.cwd) opts.cwd = body.cwd;
    term = vscode.window.createTerminal(opts);
  }

  term.show(true);  // preserve focus
  term.sendText(command);
  log(`Terminal [${name}]: ${command}`);
  json(res, 200, { success: true, terminal: name, command });
}

function handleProblems(body, res) {
  // body: { source, problems: [{ file_path, line, message, severity? }] }
  const source   = body.source || 'NsOps';
  const problems = body.problems || [];
  if (!problems.length) return json(res, 400, { error: 'problems array required' });

  // Group by file
  const byFile = {};
  problems.forEach(p => {
    if (!byFile[p.file_path]) byFile[p.file_path] = [];
    byFile[p.file_path].push(p);
  });

  const collection = vscode.languages.createDiagnosticCollection(source);
  Object.entries(byFile).forEach(([filePath, items]) => {
    const uri  = vscode.Uri.file(filePath);
    const diags = items.map(p => {
      const line  = Math.max(0, parseInt(p.line || 1) - 1);
      const range = new vscode.Range(line, 0, line, 999);
      const sev   = (p.severity || 'warning').toLowerCase();
      const sevCode = sev === 'error'   ? vscode.DiagnosticSeverity.Error
                    : sev === 'info'    ? vscode.DiagnosticSeverity.Information
                    : sev === 'hint'    ? vscode.DiagnosticSeverity.Hint
                    : vscode.DiagnosticSeverity.Warning;
      return new vscode.Diagnostic(range, p.message, sevCode);
    });
    collection.set(uri, diags);
  });

  log(`Injected ${problems.length} problem(s) from ${source}`);
  json(res, 200, { success: true, count: problems.length });
}

function handleClearHighlights(res) {
  vscode.window.visibleTextEditors.forEach(e => e.setDecorations(decorType, []));
  log('Cleared all NsOps highlights');
  json(res, 200, { success: true });
}

function handleOutput(body, res) {
  // body: { message, show? }
  const msg = body.message || '';
  outputCh.appendLine(`[NsOps] ${msg}`);
  if (body.show) outputCh.show(true);
  json(res, 200, { success: true });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function json(res, status, data) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(data));
}

function log(msg) {
  if (outputCh) outputCh.appendLine(`[${new Date().toISOString()}] ${msg}`);
  console.log(`[NsOps] ${msg}`);
}

function setStatusBar(state, port) {
  if (!statusBar) return;
  if (state === 'running') {
    statusBar.text        = `$(zap) NsOps :${port}`;
    statusBar.tooltip     = `NsOps integration server running on port ${port}\nClick for status`;
    statusBar.color       = '#22c55e';
    statusBar.backgroundColor = undefined;
  } else if (state === 'error') {
    statusBar.text        = '$(warning) NsOps Error';
    statusBar.color       = '#ef4444';
  } else {
    statusBar.text        = '$(circle-slash) NsOps';
    statusBar.tooltip     = 'NsOps integration server stopped\nRun: NsOps: Start Integration Server';
    statusBar.color       = '#6b7280';
  }
  statusBar.show();
}

function showStatus() {
  const port    = vscode.workspace.getConfiguration('nsops').get('serverPort', 6789);
  const running = !!server;
  vscode.window.showInformationMessage(
    running
      ? `NsOps server running on http://127.0.0.1:${port}`
      : 'NsOps server is stopped. Run "NsOps: Start Integration Server" to start.',
    running ? 'Stop Server' : 'Start Server'
  ).then(action => {
    if (action === 'Stop Server')  vscode.commands.executeCommand('nsops.stopServer');
    if (action === 'Start Server') vscode.commands.executeCommand('nsops.startServer');
  });
}

module.exports = { activate, deactivate };
