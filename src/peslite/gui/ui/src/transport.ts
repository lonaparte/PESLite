/** The shared frontend knows about messages, never HTTP or VS Code file APIs. */
export type Operation =
  | 'tree' | 'file.read' | 'preview' | 'csv.read' | 'summary.read'
  | 'debug.inspect'
  | 'preview.layout.load' | 'preview.layout.save' | 'preview.layout.clear'
  | 'run.start' | 'run.stop' | 'run.status'
  | 'workspace.active' | 'editor.reveal';

export type JsonObject = Record<string, unknown>;
export type ChangeEvent = { type: 'file.changed'; path: string; text: string; dirty?: boolean };
export type TreeEntry = { path: string; name: string; directory: boolean };
export type TextFile = { path: string; text: string; version: string; dirty?: boolean };
export type PreviewResponse = {
  format: 'native' | 'legacy';
  model: import('../../preview/types.js').PESModel | null;
  diagnostics: import('../../preview/types.js').Diagnostic[];
  source?: string;
};
export type CsvResponse = {
  columns: string[]; rows: (number | string | null)[][];
  nextOffset: number; nextByteOffset: number; eof: boolean;
};
export type RunStatus = { running: boolean; exitCode: number | null; path: string; log: string };
export type DebugOperation = {
  kind: 'connect' | 'disconnect' | 'set'; source: string; target: string;
  path: string; value: unknown;
};
export type DebugPreview = {
  modelPath: string; modelText: string; preview: PreviewResponse;
  operations: DebugOperation[]; sources: string[]; probes: Record<string, string>;
  analyses: string[]; checks: string[]; watch: string[];
};

interface VSCodeAPI { postMessage(message: unknown): void }
declare const acquireVsCodeApi: undefined | (() => VSCodeAPI);
const vscode = typeof acquireVsCodeApi === 'function' ? acquireVsCodeApi() : null;
export const inVSCode = vscode !== null;
let nextId = 0;
const pending = new Map<number, { resolve: (value: any) => void; reject: (error: Error) => void }>();
const listeners = new Set<(event: ChangeEvent) => void>();
const token = document.querySelector<HTMLMetaElement>('meta[name="peslite-token"]')?.content ?? '';

window.addEventListener('message', (event: MessageEvent) => {
  if (!vscode || typeof event.data !== 'object' || event.data === null) return;
  const message = event.data;
  if (message.type === 'file.changed' && typeof message.path === 'string') {
    listeners.forEach(listener => listener(message as ChangeEvent));
    return;
  }
  if (typeof message.id !== 'number') return;
  const item = pending.get(message.id);
  if (!item) return;
  pending.delete(message.id);
  if (message.ok) item.resolve(message.result);
  else item.reject(new Error(String(message.error ?? 'The host rejected the request')));
});

export function subscribe(listener: (event: ChangeEvent) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function request<T>(method: Operation, params: JsonObject = {}): Promise<T> {
  if (vscode) {
    return new Promise<T>((resolve, reject) => {
      const id = ++nextId;
      pending.set(id, { resolve, reject });
      vscode.postMessage({ id, method, params });
    });
  }
  return fetch('/api', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-PESLite-Token': token },
    body: JSON.stringify({ method, params }),
    cache: 'no-store',
  }).then(async response => {
    const data: { ok: boolean; result?: T; error?: string } = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || 'Request failed');
    return data.result as T;
  });
}

export function closePreviewSession(): void {
  if (vscode) return;
  void fetch('/api', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-PESLite-Token': token },
    body: JSON.stringify({ method: 'preview.layout.clear', params: {} }),
    cache: 'no-store', keepalive: true,
  });
}
