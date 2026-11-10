/** VS Code host adapter; the VSIX ships static UI only, never Python or scipy. */
import * as vscode from 'vscode';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as crypto from 'node:crypto';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';

interface Message { id: number; method: string; params: Record<string, any> }
interface FileEdit { start: number; end: number; replacement: string; line: number; column: number }
const safeSuffix = /\.(pes|yaml|yml|csv|json)$/i;

function python() {
  return vscode.workspace.getConfiguration('peslite').get<string>('pythonPath') || 'python';
}
function hash(value: string) { return crypto.createHash('sha256').update(value).digest('hex'); }
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
    const child = spawn(python(), ['-m', 'peslite.frontend.host', '--root', root], {
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
  start(folder: vscode.WorkspaceFolder, filename: string, trustAddons: boolean) {
    if (!vscode.workspace.isTrusted) throw new Error('Trust the VS Code workspace to run simulations');
    if (this.process) throw new Error('A simulation is already running');
    const uri = resource(folder, filename);
    if (!filename.endsWith('.pes')) throw new Error('Select a .pes model');
    const addons = [folder.uri.fsPath, path.dirname(uri.fsPath)].some(directory =>
      fs.existsSync(path.join(directory, 'PESaddons')));
    if (addons && !trustAddons) {
      throw new Error('Project-local PESaddons execute Python code; explicit trust is required');
    }
    this.output = ''; this.code = null; this.file = filename;
    const child = spawn(python(), ['-m', 'peslite.solver.simulation', uri.fsPath], {
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
  if (!target) throw new Error('Open a .pes, CSV or summary.json file first');
  const folder = folderFor(target);
  const filename = relative(folder, target);
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
        text: event.document.getText(), dirty: event.document.isDirty });
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
        result = { path: params.path, text: document.getText(),
          version: hash(document.getText()), dirty: document.isDirty };
      } else if (method === 'editor.reveal') {
        const document = await vscode.workspace.openTextDocument(resource(folder, params.path));
        const editor = await vscode.window.showTextDocument(document, { viewColumn: vscode.ViewColumn.One });
        const location = new vscode.Position(Math.max(0, Number(params.line) - 1),
          Math.max(0, Number(params.column) - 1));
        editor.selection = new vscode.Selection(location, location);
        editor.revealRange(new vscode.Range(location, location));
        result = { revealed: true };
      } else if (method === 'editor.applyPatch') {
        const document = await vscode.workspace.openTextDocument(resource(folder, params.path));
        if (document.getText() !== params.oldText) throw new Error('The document changed; retry the edit');
        const edit = params.edit as FileEdit;
        if (!(Number.isInteger(edit.start) && Number.isInteger(edit.end) && edit.start >= 0 &&
              edit.end <= document.getText().length && edit.start <= edit.end)) {
          throw new Error('Invalid source range');
        }
        const change = new vscode.WorkspaceEdit();
        const toUtf16 = (index: number) => Array.from(document.getText()).slice(0, index).join('').length;
        change.replace(document.uri,
          new vscode.Range(document.positionAt(toUtf16(edit.start)), document.positionAt(toUtf16(edit.end))),
          edit.replacement);
        if (!await vscode.workspace.applyEdit(change)) throw new Error('Could not apply the edit');
        result = { applied: true };
      } else if (method === 'preview' &&
                 String(params.text).split(/\r?\n/)
                   .find(line => line.trim() && !line.trim().startsWith('#'))
                   ?.trim().startsWith('@')) {
        result = { format: 'native', model: null, fields: [], diagnostics: [] };
      } else if (method === 'preview' || method === 'parameter.patch' ||
                 method === 'component.add' || method === 'component.remove' ||
                 method === 'connection.rewire' || method === 'csv.read' ||
                 method === 'summary.read') {
        if (params.trustAddons === true && !vscode.workspace.isTrusted) {
          throw new Error('Trust the VS Code workspace before loading project-local addons');
        }
        result = await bridge(folder.uri.fsPath, method, params);
      } else if (method === 'run.start') {
        const document = await vscode.workspace.openTextDocument(resource(folder, params.path));
        if (document.isDirty) throw new Error('Save the .pes document before running a simulation');
        result = runs.start(folder, params.path, params.trustAddons === true);
      } else if (method === 'run.stop') result = runs.stop();
      else if (method === 'run.status') result = runs.status();
      else throw new Error(`Unsupported VS Code operation: ${method}`);
      await panel.webview.postMessage({ id, ok: true, result });
    } catch (problem) {
      await panel.webview.postMessage({ id, ok: false,
        error: problem instanceof Error ? problem.message : String(problem) });
    }
  }));
  panel.onDidDispose(() => disposables.forEach(disposable => disposable.dispose()));
}

export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(vscode.commands.registerCommand('peslite.preview', async uri => {
    try { await openPreview(context, uri instanceof vscode.Uri ? uri : undefined); }
    catch (error) { void vscode.window.showErrorMessage(String(error)); }
  }));
  context.subscriptions.push(vscode.commands.registerCommand('peslite.run', async () => {
    const uri = vscode.window.activeTextEditor?.document.uri;
    if (!uri) return;
    try { runs.start(folderFor(uri), relative(folderFor(uri), uri), false); }
    catch (error) { void vscode.window.showErrorMessage(String(error)); }
  }));
  context.subscriptions.push(vscode.commands.registerCommand('peslite.stop', () => runs.stop()));
}

export function deactivate() { runs.stop(); }
