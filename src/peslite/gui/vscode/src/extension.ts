/** VS Code host adapter; the VSIX ships static UI only, never Python or scipy. */
import * as vscode from 'vscode';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';

interface Message { id: number; method: string; params: Record<string, any> }
const bridged = new Set(['preview', 'debug.inspect', 'csv.read', 'summary.read', 'json.read']);
const safeSuffix = /\.(pes|pesdbg|yaml|yml|csv|json)$/i;

function python() {
  return vscode.workspace.getConfiguration('peslite').get<string>('pythonPath') || 'python';
}
function folderFor(uri: vscode.Uri): vscode.WorkspaceFolder {
  const folder = vscode.workspace.getWorkspaceFolder(uri);
  if (!folder || uri.scheme !== 'file') throw new Error('Open a file inside a trusted workspace');
  return folder;
}
function relative(folder: vscode.WorkspaceFolder, uri: vscode.Uri): string {
  const value = path.relative(folder.uri.fsPath, uri.fsPath).replace(/\\/g, '/');
  if (!value || value.startsWith('../') || value === '..' || path.isAbsolute(value) ||
      value.split('/').some(segment => segment.startsWith('.'))) {
    throw new Error('The selected path is outside the workspace');
  }
  return value;
}
function resource(folder: vscode.WorkspaceFolder, filename: string): vscode.Uri {
  if (!filename || filename.includes('\\') || filename.split('/').some(segment =>
      !segment || segment === '.' || segment === '..' || segment.startsWith('.'))) {
    throw new Error('Invalid workspace-relative file path');
  }
  if (!safeSuffix.test(filename)) throw new Error('Unsupported file type');
  const uri = vscode.Uri.file(path.join(folder.uri.fsPath, ...filename.split('/')));
  relative(folder, uri);
  // A symlink inside the workspace may otherwise resolve outside it.
  const realRoot = fs.realpathSync(folder.uri.fsPath);
  const realFile = fs.realpathSync(uri.fsPath);
  if (!realFile.startsWith(realRoot + path.sep)) {
    throw new Error('Symlink target is outside the workspace');
  }
  return uri;
}

async function bridge(root: string, method: string, params: Record<string, unknown>): Promise<any> {
  return new Promise((resolve, reject) => {
    const child = spawn(python(), ['-m', 'peslite.gui.host', '--root', root], {
      cwd: root, env: process.env, stdio: 'pipe', windowsHide: true,
    });
    let output = '';
    let error = '';
    const timeout = setTimeout(() => child.kill(), 20000);
    child.stdout.on('data', chunk => { output += String(chunk); });
    child.stderr.on('data', chunk => { error += String(chunk); });
    child.on('error', err => { clearTimeout(timeout); reject(err); });
    child.on('close', code => {
      clearTimeout(timeout);
      try {
        const payload = JSON.parse(output);
        if (!payload.ok) reject(new Error(payload.error || 'PESLite rejected the operation'));
        else resolve(payload.result);
      } catch (err) {
        reject(new Error(`PESLite Python bridge failed (exit ${code}): ${error.slice(-1000)}`));
      }
    });
    child.stdin.end(JSON.stringify({ method, params }));
  });
}

class Runs {
  private process: ChildProcessWithoutNullStreams | null = null;
  private output = '';
  private file = '';
  private code: number | null = null;

  status() {
    return { running: this.process !== null, exitCode: this.code,
      path: this.file, log: this.output.slice(-30000) };
  }
  async start(folder: vscode.WorkspaceFolder, filename: string, trustAddons: boolean) {
    if (!vscode.workspace.isTrusted) throw new Error('Trust the VS Code workspace to run simulations');
    if (this.process) throw new Error('A simulation is already running');
    resource(folder, filename);
    const { command }: { command: string[] } =
      await bridge(folder.uri.fsPath, 'run.command', { path: filename, trustAddons });
    if (this.process) throw new Error('A simulation is already running');
    this.output = ''; this.code = null; this.file = filename;
    const child = spawn(command[0], command.slice(1), {
      cwd: folder.uri.fsPath, env: process.env, windowsHide: true,
    });
    this.process = child;
    const append = (value: Buffer) => { this.output = (this.output + String(value)).slice(-30000); };
    child.stdout.on('data', append);
    child.stderr.on('data', append);
    child.on('error', err => {
      this.output += '\n' + err.message;
      this.code = 1; this.process = null;
    });
    child.on('close', code => { this.code = code ?? 1; this.process = null; });
    return this.status();
  }
  stop() { this.process?.kill(); return { stopping: !!this.process }; }
}
const runs = new Runs();
const layouts = new Map<string, unknown>();

function webviewHtml(panel: vscode.WebviewPanel, context: vscode.ExtensionContext): string {
  const media = vscode.Uri.joinPath(context.extensionUri, 'media');
  const index = fs.readFileSync(path.join(media.fsPath, 'index.html'), 'utf8');
  const csp = `default-src 'none'; img-src ${panel.webview.cspSource} data:; ` +
    `script-src ${panel.webview.cspSource}; style-src ${panel.webview.cspSource} 'unsafe-inline'; ` +
    `font-src ${panel.webview.cspSource}; connect-src 'none'`;
  return index.replace('</head>', `<meta http-equiv="Content-Security-Policy" content="${csp}"></head>`)
    .replace(/(src|href)="\.\/([^"]+)"/g, (_match, attr: string, filename: string) =>
      `${attr}="${panel.webview.asWebviewUri(vscode.Uri.joinPath(media, ...filename.split('/')))}"`)
    .replace('__PESLITE_TOKEN__', '');
}

async function openPreview(context: vscode.ExtensionContext, uri?: vscode.Uri) {
  const target = uri || vscode.window.activeTextEditor?.document.uri;
  if (!target) throw new Error('Open a .pes, .pesdbg, CSV or summary.json file first');
  const folder = folderFor(target);
  const filename = relative(folder, target);
  const layoutKey = `${folder.uri.toString()}\0${filename}`;
  if (!safeSuffix.test(filename)) throw new Error('Unsupported file type');
  const panel = vscode.window.createWebviewPanel('peslite.preview', `PESLite: ${path.basename(filename)}`,
    vscode.ViewColumn.Beside, {
      enableScripts: true,
      retainContextWhenHidden: true,
      localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'media')],
    });
  panel.webview.html = webviewHtml(panel, context);
  const disposables: vscode.Disposable[] = [];
  disposables.push(vscode.workspace.onDidChangeTextDocument(event => {
    if (event.document.uri.toString() === target.toString()) {
      void panel.webview.postMessage({ type: 'file.changed', path: filename,
        text: event.document.getText() });
    }
  }));
  disposables.push(panel.webview.onDidReceiveMessage(async (message: Message) => {
    const { method, params, id } = message;
    try {
      let result: unknown;
      if (method === 'workspace.active') {
        result = { path: filename };
      } else if (method === 'file.read') {
        const document = await vscode.workspace.openTextDocument(resource(folder, params.path));
        result = { path: params.path, text: document.getText() };
      } else if (method === 'editor.reveal') {
        const document = await vscode.workspace.openTextDocument(resource(folder, params.path));
        const editor = await vscode.window.showTextDocument(document, { viewColumn: vscode.ViewColumn.One });
        const location = new vscode.Position(Math.max(0, Number(params.line) - 1),
          Math.max(0, Number(params.column) - 1));
        editor.selection = new vscode.Selection(location, location);
        editor.revealRange(new vscode.Range(location, location));
        result = { revealed: true };
      } else if (method === 'preview.layout.load') {
        result = { layout: layouts.get(layoutKey) ?? { format: 'peslite.preview-layout', version: 1, views: {} } };
      } else if (method === 'preview.layout.save') {
        layouts.set(layoutKey, params.layout);
        result = { layout: params.layout };
      } else if (method === 'preview.layout.clear') {
        layouts.delete(layoutKey);
        result = { cleared: true };
      } else if (bridged.has(method)) {
        if (params.trustAddons === true && !vscode.workspace.isTrusted) {
          throw new Error('Trust the VS Code workspace before loading project-local addons');
        }
        result = await bridge(folder.uri.fsPath, method, params);
      } else if (method === 'run.start') {
        const document = await vscode.workspace.openTextDocument(resource(folder, params.path));
        if (document.isDirty) throw new Error('Save the .pes document before running a simulation');
        result = await runs.start(folder, params.path, params.trustAddons === true);
      } else if (method === 'run.stop') result = runs.stop();
      else if (method === 'run.status') result = runs.status();
      else throw new Error(`Unsupported VS Code operation: ${method}`);
      await panel.webview.postMessage({ id, ok: true, result });
    } catch (problem) {
      await panel.webview.postMessage({ id, ok: false,
        error: problem instanceof Error ? problem.message : String(problem) });
    }
  }));
  panel.onDidDispose(() => {
    layouts.delete(layoutKey);
    disposables.forEach(disposable => disposable.dispose());
  });
}

export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(vscode.commands.registerCommand('peslite.preview', async uri => {
    try { await openPreview(context, uri instanceof vscode.Uri ? uri : undefined); }
    catch (error) { void vscode.window.showErrorMessage(String(error)); }
  }));
  context.subscriptions.push(vscode.commands.registerCommand('peslite.run', async () => {
    const document = vscode.window.activeTextEditor?.document;
    if (!document) return;
    try {
      if (document.isDirty) throw new Error('Save the document before running a simulation');
      const uri = document.uri;
      await runs.start(folderFor(uri), relative(folderFor(uri), uri), false);
    }
    catch (error) { void vscode.window.showErrorMessage(String(error)); }
  }));
  context.subscriptions.push(vscode.commands.registerCommand('peslite.stop', () => runs.stop()));
}

export function deactivate() { runs.stop(); }
