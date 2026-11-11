import { useCallback, useEffect, useRef, useState } from 'react';
import { createPESPreview, parsePES } from '../../preview/index.js';
import type { GraphEdge, GraphNode, PESModel, SourceLocation } from '../../preview/types.js';
import { request, subscribe, inVSCode } from './transport';
import type { Operation } from './transport';
import type {
  CsvResponse, Field, PreviewResponse, RunStatus, SourceEdit, TextFile, TreeEntry,
} from './transport';
import { SourceEditor } from './SourceEditor';
import { Waveform } from './Waveform';

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
function utf16Offset(text: string, pythonIndex: number): number {
  return Array.from(text).slice(0, pythonIndex).join('').length;
}
function parseScalar(input: string): unknown {
  if (input.trim() === 'true') return true;
  if (input.trim() === 'false') return false;
  if (input.trim() === 'null') return null;
  if (input.trim() && Number.isFinite(Number(input))) return Number(input);
  try { return JSON.parse(input) as unknown; } catch { return input; }
}

function FileTree({ directory, current, onOpen, stamp }: {
  directory: string; current: string; onOpen: (path: string) => void; stamp: number;
}) {
  const [entries, setEntries] = useState<TreeEntry[]>([]);
  const [expanded, setExpanded] = useState<string[]>([]);
  useEffect(() => {
    let mounted = true;
    request<{ files: TreeEntry[] }>('tree', { path: directory })
      .then(data => { if (mounted) setEntries(data.files); })
      .catch(() => { if (mounted) setEntries([]); });
    return () => { mounted = false; };
  }, [directory, stamp]);
  return <div className="tree-list">
    {entries.map(entry => <div key={entry.path}>
      <button className={'tree-entry ' + (entry.path === current ? 'active' : '')}
        onClick={() => entry.directory ? setExpanded(old => old.includes(entry.path)
          ? old.filter(item => item !== entry.path) : [...old, entry.path]) : onOpen(entry.path)}>
        <span aria-hidden="true">{entry.directory ? expanded.includes(entry.path) ? '▾' : '▸' : '◇'}</span>
        {entry.name}
      </button>
      {entry.directory && expanded.includes(entry.path) &&
        <FileTree directory={entry.path} current={current} onOpen={onOpen} stamp={stamp} />}
    </div>)}
  </div>;
}

function GraphPane({ text, path, response, onSelect, onSource }: {
  text: string; path: string; response: PreviewResponse | null;
  onSelect: (node: GraphNode | GraphEdge, view: string, kind: 'node' | 'edge') => void;
  onSource: (location: SourceLocation) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const preview = useRef<ReturnType<typeof createPESPreview> | null>(null);
  const callback = useRef({ onSelect, onSource });
  callback.current = { onSelect, onSource };
  useEffect(() => {
    if (!container.current) return;
    preview.current = createPESPreview(container.current, {
      onNodeSelect: (node, view) => callback.current?.onSelect(node, view, 'node'),
      onEdgeSelect: (edge, view) => callback.current?.onSelect(edge, view, 'edge'),
      onSourceSelect: loc => callback.current?.onSource(loc),
    });
    return () => { preview.current?.destroy(); preview.current = null; };
  }, []);
  useEffect(() => {
    if (!preview.current || !response) return;
    if (response.format === 'legacy') {
      if (response.model) preview.current.updateModel(response.model);
      else preview.current.updateModel({
        views: [], globals: [], timeline: [], events: [], defaultView: 'circuit',
        diagnostics: response.diagnostics,
      });
    } else {
      const model = parsePES(text);
      const host = response.source === text ? response.diagnostics : [];
      const lines = new Set(model.diagnostics.map(item => item.range.start.line));
      preview.current.updateModel({
        ...model,
        diagnostics: [...model.diagnostics, ...host.filter(item => !lines.has(item.range.start.line))],
      });
    }
  }, [text, path, response]);
  return <div className="graph-container" ref={container} />;
}

type Selection = { name: string; id: string; view: string; kind: 'node' | 'edge' };

export function App() {
  const [path, setPath] = useState('');
  const [file, setFile] = useState<TextFile | null>(null);
  const [text, setText] = useState('');
  const [csv, setCsv] = useState<CsvResponse | null>(null);
  const [comparison, setComparison] = useState<{ path: string; data: CsvResponse } | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [summary, setSummary] = useState<{ summary: unknown; simulationFile: string | null } | null>(null);
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [selected, setSelected] = useState<Selection | null>(null);
  const [fields, setFields] = useState<Record<string, string>>({});
  const [jumpLine, setJumpLine] = useState<number | null>(null);
  const [tab, setTab] = useState<'split' | 'source' | 'diagram'>('split');
  const [error, setError] = useState('');
  const [stamp, setStamp] = useState(0);
  const [run, setRun] = useState<RunStatus | null>(null);
  const [stride, setStride] = useState(1);
  const [showOutput, setShowOutput] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [trustAddons, setTrustAddons] = useState(false);
  const activePath = useRef(path);
  activePath.current = path;
  const currentText = useRef(text);
  currentText.current = text;
  const kind = path.toLowerCase().endsWith('.csv') ? 'csv'
    : path.toLowerCase().endsWith('summary.json') ? 'summary'
    : path.toLowerCase().endsWith('.pes') ? 'pes' : 'text';
  const isPes = !!path && kind === 'pes';

  const open = useCallback(async (name: string) => {
    try {
      setError('');
      const selectedPath = name || (await request<{ path: string }>('workspace.active')).path;
      if (!selectedPath) return;
      setPath(selectedPath); setSelected(null); setFields({}); setPreview(null);
      setShowOutput(false); setComparison(null);
      if (selectedPath.endsWith('.csv')) {
        setCsv(await request<CsvResponse>('csv.read', {
          path: selectedPath, offset: 0, stride: 1, limit: 1200,
        }));
        setStride(1); setSummary(null); setFile(null); setText(''); setDirty(false);
      } else if (selectedPath.endsWith('summary.json')) {
        setSummary(await request<{ summary: unknown; simulationFile: string | null }>(
          'summary.read', { path: selectedPath }));
        setCsv(null); setFile(null); setText(''); setDirty(false);
      } else {
        const content = await request<TextFile>('file.read', { path: selectedPath });
        setFile(content); setText(content.text); setDirty(!!content.dirty);
        setCsv(null); setSummary(null);
      }
    } catch (problem) { setError(message(problem)); }
  }, []);

  useEffect(() => {
    if (inVSCode) void open('');
    return subscribe(event => {
      if (event.path === activePath.current) {
        setText(event.text);
        setDirty(!!event.dirty);
      }
    });
  }, [open]);

  useEffect(() => {
    if (!isPes || !file) { setPreview(null); return; }
    let active = true;
    const timer = setTimeout(() => {
      request<PreviewResponse>('preview', { path, text, trustAddons })
        .then(result => { if (active) setPreview(result); })
        .catch(problem => { if (active) setError(message(problem)); });
    }, 260);
    return () => { active = false; clearTimeout(timer); };
  }, [isPes, file?.path, text, trustAddons]);

  useEffect(() => {
    const poll = () => request<RunStatus>('run.status').then(setRun).catch(() => {});
    poll();
    const timer = setInterval(poll, 1600);
    return () => clearInterval(timer);
  }, []);

  // Continue reading existing or growing files even when the simulator runs
  // in another process. Byte cursors avoid rescanning previously plotted rows.
  useEffect(() => {
    if (!path.toLowerCase().endsWith('.csv') || !csv || !autoRefresh) return;
    let active = true;
    const timer = setInterval(() => {
      request<CsvResponse>('csv.read', {
        path, offset: csv.nextOffset, byteOffset: csv.nextByteOffset,
        stride, limit: 1200,
      }).then(next => {
        if (!active) return;
        setCsv(current => current && current.nextByteOffset === csv.nextByteOffset &&
          current.nextOffset === csv.nextOffset &&
          (next.rows.length || next.nextByteOffset !== current.nextByteOffset)
          ? { ...next, rows: [...current.rows, ...next.rows].slice(-8000) } : current);
      }).catch(() => {});
    }, 1500);
    return () => { active = false; clearInterval(timer); };
  }, [path, csv?.nextByteOffset, csv?.nextOffset, stride, autoRefresh]);

  useEffect(() => {
    if (!comparison || !autoRefresh) return;
    let active = true;
    const timer = setInterval(() => {
      const previous = comparison;
      request<CsvResponse>('csv.read', {
        path: previous.path, offset: previous.data.nextOffset,
        byteOffset: previous.data.nextByteOffset, stride, limit: 1200,
      }).then(next => {
        if (!active) return;
        setComparison(current => current && current.path === previous.path &&
          current.data.nextByteOffset === previous.data.nextByteOffset &&
          (next.rows.length || next.nextByteOffset !== current.data.nextByteOffset)
          ? { path: current.path,
              data: { ...next, rows: [...current.data.rows, ...next.rows].slice(-8000) } }
          : current);
      }).catch(() => {});
    }, 1500);
    return () => { active = false; clearInterval(timer); };
  }, [comparison?.path, comparison?.data.nextByteOffset, comparison?.data.nextOffset,
      autoRefresh, stride]);

  async function save() {
    if (!file || inVSCode) return;
    try {
      const saved = await request<{ version: string }>('file.write', {
        path: file.path, text, version: file.version,
      });
      setFile({ ...file, version: saved.version, text }); setDirty(false);
      setStamp(v => v + 1);
    } catch (problem) { setError(message(problem)); }
  }

  async function applySourceEdit(method: Operation, params: Record<string, unknown>) {
    if (!file || !isPes) return;
    if (!inVSCode && (file.text !== text || dirty)) {
      setError('Save pending source edits before applying a schematic change.');
      return;
    }
    try {
      const { edit } = await request<{ edit: SourceEdit }>(method, {
        path, text, trustAddons, ...params,
      });
      if (currentText.current !== text) {
        throw new Error('Source changed while validating the edit; retry the operation.');
      }
      const start = utf16Offset(text, edit.start);
      const end = utf16Offset(text, edit.end);
      const updated = text.slice(0, start) + edit.replacement + text.slice(end);
      if (inVSCode) {
        await request('editor.applyPatch', { path, oldText: text, edit });
      } else {
        const saved = await request<{ version: string }>('file.write', {
          path, text: updated, version: file.version,
        });
        setFile({ ...file, version: saved.version, text: updated });
        setText(updated); setDirty(false);
      }
      setFields({});
      setError('');
    } catch (problem) { setError(message(problem)); }
  }

  async function applyParameter(field: Field) {
    await applySourceEdit('parameter.patch', { key: field.path,
      value: parseScalar(fields[field.path] ?? text.slice(
        utf16Offset(text, field.start), utf16Offset(text, field.end))) });
  }

  async function addSchematicComponent() {
    if (!isPes || preview?.format !== 'legacy') return;
    const section = window.prompt('Component section: buses, sources, branches, elements, units',
      'buses');
    if (!section) return;
    const name = window.prompt('New component name (letters, numbers, underscore)', 'new_bus');
    if (!name) return;
    const definition = window.prompt('YAML mapping for the new component (validated by PESLite)',
      section === 'buses' ? '{phases: 3, c_pu: 0.02}' :
      section === 'elements' ? '{type: resistor, bus: pcc, r: 1.0}' : '{}');
    if (definition === null) return;
    await applySourceEdit('component.add', { section, name, definition });
  }

  async function removeSchematicComponent(section: string, name: string) {
    if (!window.confirm(`Remove ${section}.${name}? The operation will be rejected if unsafe.`)) return;
    await applySourceEdit('component.remove', { section, name });
    setSelected(null);
  }

  async function rewire(field: Field, buses: string[]) {
    const oldValue = text.slice(utf16Offset(text, field.start), utf16Offset(text, field.end));
    const nextBus = window.prompt(`Reconnect ${field.path} to a bus:\n${buses.join(', ')}`,
      oldValue.replace(/^['"]|['"]$/g, ''));
    if (!nextBus) return;
    await applySourceEdit('connection.rewire', { key: field.path, target: nextBus });
  }

  async function act(action: 'file.create' | 'file.rename' | 'file.delete') {
    if (inVSCode) return;
    let payload: Record<string, unknown>;
    if (action === 'file.create') {
      const name = window.prompt('New file path:', 'new-case.pes');
      if (!name) return;
      payload = { path: name, text: '' };
    } else {
      if (!file) return;
      if (action === 'file.delete') {
        if (!window.confirm('Delete ' + path + '?')) return;
        payload = { path, version: file.version };
      } else {
        const destination = window.prompt('Rename file to:', path);
        if (!destination || destination === path) return;
        payload = { source: path, destination, version: file.version };
      }
    }
    try {
      await request(action, payload);
      setStamp(v => v + 1);
      if (action === 'file.create') void open(payload.path as string);
      else if (action === 'file.rename') void open(payload.destination as string);
      else { setFile(null); setPath(''); setPreview(null); }
    } catch (problem) { setError(message(problem)); }
  }

  async function loadCsv(more: boolean, nextStride = stride) {
    try {
      const response = await request<CsvResponse>('csv.read', {
        path, offset: more ? csv?.nextOffset || 0 : 0,
        byteOffset: more ? csv?.nextByteOffset : undefined,
        stride: nextStride, limit: 1200,
      });
      setCsv(more && csv ? { ...response, rows: [...csv.rows, ...response.rows].slice(-8000) }
        : response);
      setStride(nextStride);
      if (comparison) {
        const other = await request<CsvResponse>('csv.read', {
          path: comparison.path, offset: 0, limit: 1200, stride: nextStride,
        });
        setComparison({ path: comparison.path, data: other });
      }
    } catch (problem) { setError(message(problem)); }
  }

  async function compareCsv() {
    if (!csv) return;
    const candidate = window.prompt('CSV file to compare (workspace-relative path)',
      path.replace(/[^/]*$/, 'plant.csv'));
    if (!candidate) return;
    try {
      const other = await request<CsvResponse>('csv.read', {
        path: candidate, offset: 0, stride, limit: 1200,
      });
      setComparison({ path: candidate, data: other });
      setError('');
    } catch (problem) { setError(message(problem)); }
  }

  async function execute() {
    if (!isPes) return;
    if (dirty || (file && text !== file.text)) {
      setError('Save your .pes source before running the CLI.'); return;
    }
    try {
      setError(''); setShowOutput(true);
      setRun(await request<RunStatus>('run.start', { path, trustAddons }));
    } catch (problem) {
      if (message(problem).includes('explicit trust') && window.confirm(
        'PESaddons contains executable project Python. Trust this project and run it?')) {
        try { setTrustAddons(true); setRun(await request<RunStatus>('run.start', { path, trustAddons: true })); }
        catch (retry) { setError(message(retry)); }
      } else setError(message(problem));
    }
  }

  async function reveal(location: SourceLocation) {
    if (inVSCode) {
      try { await request('editor.reveal', { path, line: location.line, column: location.column }); }
      catch (problem) { setError(message(problem)); }
    } else { setJumpLine(location.line); setTab('split'); }
  }

  const model: PESModel | null = preview?.model ?? null;
  const unitPrefix = selected && model ? (() => {
    if (selected.view !== 'circuit' && selected.kind === 'node')
      return 'units.' + selected.view + '.ctrl.loops.' + selected.name + '.';
    const id = selected.id;
    // A branch or two-terminal element is represented as a connection, not
    // as a node in the PESLite topology descriptor.
    if (selected.kind === 'edge' && id.startsWith('branch-')) {
      return 'branches.' + id.slice('branch-'.length) + '.';
    }
    if (selected.kind === 'edge' && id.startsWith('element-') &&
        !id.endsWith('-terminal')) {
      return 'elements.' + id.slice('element-'.length) + '.';
    }
    const owner = selected.kind === 'edge'
      ? model.views[0]?.edges.find(edge => edge.id === id) : null;
    const associated = owner
      ? [owner.from, owner.to].find(value => /^(unit|source|element)-/.test(value))
      : id;
    if (!associated) return '';
    for (const [prefix, section] of [
      ['bus-', 'buses'], ['unit-', 'units'], ['source-', 'sources'],
      ['element-', 'elements'], ['branch-', 'branches'],
    ]) {
      if (associated.startsWith(prefix)) {
        const component = model.views[0]?.nodes.find(node => node.id === associated);
        const rawName = component?.name ?? (associated === id
          ? selected.name : associated.slice(prefix.length));
        return section + '.' + rawName + '.';
      }
    }
    return '';
  })() : '';
  const editable = (preview?.fields || []).filter(field =>
    Boolean(unitPrefix) && field.path.startsWith(unitPrefix)).slice(0, 80);
  const buses = (model?.views.find(view => view.name === 'circuit')?.nodes ?? [])
    .filter(node => node.id.startsWith('bus-')).map(node => node.name);
  const selectedSection = unitPrefix.split('.')[0];
  const selectedName = unitPrefix.slice(selectedSection.length + 1).replace(/\.$/, '');

  return <div className="app">
    <header className="top-bar">
      <div className="identity"><span className="logo">P</span><strong>PESLite</strong><span>Workspace</span></div>
      <div className="header-actions">
        {isPes && <><button title="Load project-local Python addons only after explicit trust"
          onClick={() => { if (window.confirm('Allow this workspace to execute local PESaddons Python code during preview and simulation?')) setTrustAddons(true); }}
          disabled={trustAddons}>{trustAddons ? 'Addons trusted' : 'Trust addons'}</button>
          <button onClick={execute} disabled={!!run?.running}>▶ Run</button>
          <button disabled={!run?.running} onClick={() =>
            request('run.stop').then(() => setShowOutput(true)).catch(problem => setError(message(problem)))}>■ Stop</button></>}
        {file && !inVSCode && <button className="primary" disabled={!dirty && file.text === text}
          onClick={save}>Save{dirty ? ' •' : ''}</button>}
      </div>
    </header>
    <div className="workspace-body">
      {!inVSCode && <aside className="sidebar">
        <div className="sidebar-title"><strong>FILES</strong>
          <button onClick={() => setStamp(v => v + 1)} title="Refresh">↻</button></div>
        <FileTree directory="" current={path} onOpen={value => void open(value)} stamp={stamp} />
        <div className="file-actions">
          <button onClick={() => void act('file.create')}>New</button>
          <button disabled={!file} onClick={() => void act('file.rename')}>Rename</button>
          <button disabled={!file} onClick={() => void act('file.delete')}>Delete</button>
        </div>
      </aside>}
      <main className="content">
        <div className="document-bar">
          <strong>{path || (inVSCode ? 'Select a .pes document in the editor' : 'Select a file')}</strong>
          {isPes && !inVSCode && <div className="tabs">{(['diagram', 'source', 'split'] as const)
            .map(value => <button key={value} className={tab === value ? 'active' : ''}
              onClick={() => setTab(value)}>{value}</button>)}</div>}
        </div>
        {error && <div className="error-bar" role="alert">{error}
          <button onClick={() => setError('')}>Dismiss</button></div>}
        {isPes && file && <>
          {preview?.format === 'legacy' && <div className="schematic-actions">
            <button onClick={() => void addSchematicComponent()}>Add component</button>
            <span>Validated source-range edits only; unsafe changes are rejected.</span>
          </div>}
          <div className={'editor-layout ' + (inVSCode ? 'only-diagram' : tab)}>
            {(inVSCode || tab !== 'source') && <GraphPane path={path} text={text}
              response={preview} onSelect={(entity, view, selectedKind) => setSelected({
                name: selectedKind === 'node' ? (entity as GraphNode).name : entity.id,
                id: entity.id, view, kind: selectedKind,
              })}
              onSource={location => void reveal(location)} />}
            {!inVSCode && tab !== 'diagram' && <section className="source-pane">
              <h3>Source{dirty && <span className="dirty"> · unsaved</span>}</h3>
              <SourceEditor key={path} text={text} onChange={value => {
                setText(value); setDirty(true);
              }} jumpLine={jumpLine} />
            </section>}
          </div>
          {selected && <section className="inspector">
            <div className="section-header"><h3>{selected.name} · {selected.kind === 'edge' ? 'connection' : 'parameters'}</h3>
              <div className="inspector-buttons">
                {preview?.format === 'legacy' && selected.kind === 'node' &&
                  ['buses', 'units', 'elements', 'sources', 'branches'].includes(selectedSection) &&
                  !selectedName.includes('.') && <button onClick={() =>
                    void removeSchematicComponent(selectedSection, selectedName)}>Remove</button>}
                <button onClick={() => setSelected(null)}>Close</button>
              </div></div>
            {!editable.length && <p className="hint">No safe scalar parameters. Edit the source text instead.</p>}
            <div className="fields">{editable.map(field => <label key={field.path}>
              <span>{field.path.slice(unitPrefix.length)}</span>
              <input aria-label={field.path} value={fields[field.path] ?? text.slice(
                utf16Offset(text, field.start), utf16Offset(text, field.end))}
                onChange={(event: { target: { value: string } }) => setFields(old => ({ ...old, [field.path]: event.target.value }))} />
              <button onClick={() => void applyParameter(field)}>Apply</button>
              {preview?.format === 'legacy' && /\.(bus|bus1|bus2)$/.test(field.path) &&
                buses.length > 0 && <button title="Reconnect a terminal to an existing bus"
                  onClick={() => void rewire(field, buses)}>Reconnect</button>}
            </label>)}</div>
          </section>}
        </>}
        {!!path && kind === 'text' && file && !inVSCode && <section className="source-pane">
          <h3>Source{dirty && <span className="dirty"> · unsaved</span>}</h3>
          <SourceEditor key={path} text={text} onChange={value => { setText(value); setDirty(true); }}
            jumpLine={jumpLine} />
        </section>}
        {!!path && kind === 'csv' && csv && <section className="results">
          <div className="result-controls"><strong>{path}</strong>
            <label>Downsample <select value={stride} onChange={(event: { target: { value: string } }) => void loadCsv(false, Number(event.target.value))}>
              {[1, 10, 100, 1000].map(value => <option key={value} value={value}>1 in {value}</option>)}
            </select></label>
            <button onClick={() => void loadCsv(false)}>Refresh</button>
            <label><input type="checkbox" checked={autoRefresh}
              onChange={(event: { target: { checked: boolean } }) => setAutoRefresh(event.target.checked)} />Auto-refresh</label>
            <button disabled={csv.eof} onClick={() => void loadCsv(true)}>More rows</button>
            <button onClick={() => void compareCsv()}>Compare CSV</button>
            {comparison && <button onClick={() => setComparison(null)}>Remove comparison</button>}
          </div>
          <Waveform key={path} data={csv} comparison={comparison} />
        </section>}
        {!!path && kind === 'summary' && summary && <section className="results">
          <h3>Simulation summary</h3>
          {summary.simulationFile && <button onClick={() => void open(summary.simulationFile!)}>
            Open simulation.pes</button>}
          <pre>{JSON.stringify(summary.summary, null, 2)}</pre>
        </section>}
        {!path && <div className="welcome"><h2>File-backed simulation workspace</h2>
          <p>Select a .pes file to edit the original YAML or inspect its circuit. CSV and JSON results can be opened without rerunning.</p></div>}
        {run && (run.running || run.log || showOutput) && <section className="run-output">
          <div className="section-header"><h3>CLI · {run.running ? 'running' : `exit ${run.exitCode ?? '—'}`}</h3>
            <button onClick={() => setShowOutput(!showOutput)}>{showOutput ? 'Hide' : 'Show'}</button></div>
          {showOutput && <pre>{run.log || 'Starting PESLite...'}</pre>}
        </section>}
      </main>
    </div>
  </div>;
}
