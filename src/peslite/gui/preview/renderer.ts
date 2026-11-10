import { parsePES } from './parser.js';
import { layoutView, normalizeLayout } from './layout.js';
import { chooseSides, pathData, routeEdge } from './routing.js';
import { niceTicks, segmentAt } from './timeline.js';
import type { Anchor, Box, Side } from './routing.js';
import type {
  Diagnostic, GraphGroup, GraphLayout, GraphView, LayoutNode, NodeOffset, PESModel,
  PreviewLayout, SourceLocation,
} from './types.js';

const NS = 'http://www.w3.org/2000/svg';
const GRID = 8;
const ALIGN_DISTANCE = 8;
const DRAG_THRESHOLD = 3;
let instance = 0;

export interface PESPreviewOptions {
  onSourceSelect?: (location: SourceLocation) => void;
  onDiagnostics?: (diagnostics: readonly Diagnostic[]) => void;
  layout?: unknown;
  onLayoutChange?: (layout: PreviewLayout) => void;
  onTimeChange?: (time: number) => void;
  title?: string;
}
export interface PESPreviewInstance {
  update(text: string): PESModel;
  navigate(viewName: string): boolean;
  getModel(): PESModel;
  getLayout(): PreviewLayout;
  setLayout(layout: unknown): void;
  getTime(): number;
  setTime(time: number): void;
  setTitle(title: string): void;
  destroy(): void;
}

interface Endpoint { key: string; group: boolean; point: LayoutNode }
interface DragState {
  pointer: number; startX: number; startY: number; ids: string[];
  base: Map<string, NodeOffset>; positions: Map<string, { x: number; y: number }>;
  scale: number; moved: boolean;
}
interface PanState { pointer: number; x: number; y: number; panX: number; panY: number }

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string): HTMLElementTagNameMap[K] {
  const result = document.createElement(tag);
  if (className) result.className = className;
  return result;
}
function svg<K extends keyof SVGElementTagNameMap>(tag: K,
  attributes: Record<string, string> = {}): SVGElementTagNameMap[K] {
  const result = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attributes)) result.setAttribute(key, value);
  return result;
}
function button(label: string, action: () => void, className?: string): HTMLButtonElement {
  const result = element('button', className);
  result.type = 'button';
  result.textContent = label;
  result.addEventListener('click', action);
  return result;
}
function formatTime(value: number): string { return Number(value.toPrecision(5)).toString(); }
function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

const STYLES = `
.pes-preview { font:13px/1.5 system-ui,sans-serif; color:#24384a; background:#fff;
  border:1px solid #d5e0eb; border-radius:9px; overflow:hidden; }
.pes-preview * { box-sizing:border-box; }
.pes-preview button { font:inherit; cursor:pointer; color:inherit; border:1px solid #cbd8e4;
  background:#fff; border-radius:5px; padding:4px 9px; }
.pes-preview button:hover { border-color:#4d90be; }
.pes-preview button:focus-visible,.pes-preview summary:focus-visible,.pes-preview .pp-axis:focus-visible {
  outline:2px solid #287aaa; }
.pes-preview .pp-title { padding:9px 12px 0; background:#f5f9fc; font-weight:700; font-size:14px;
  color:#174564; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.pes-preview .pp-toolbar { display:flex; gap:10px; flex-wrap:nowrap; align-items:center; padding:8px 12px;
  border-bottom:1px solid #e1e9f1; background:#f5f9fc; }
.pes-preview .pp-strip { display:flex; align-items:center; gap:4px; min-width:0; }
.pes-preview .pp-views { flex:1 1 0; }
.pes-preview .pp-tabs { display:flex; gap:6px; min-width:0; overflow-x:auto; scrollbar-width:none; }
.pes-preview .pp-tabs::-webkit-scrollbar { display:none; }
.pes-preview .pp-tabs button { flex:0 0 auto; }
.pes-preview .pp-tools { display:flex; gap:7px; margin-left:auto; flex:0 0 auto; }
.pes-preview .pp-arrow { padding:4px 7px; }
.pes-preview .pp-active { color:#065e98!important; border-color:#76aed0!important; background:#e4f2fb!important; }
.pes-preview .pp-timeline { padding:8px 12px 5px; border-bottom:1px solid #e1e9f1; background:#fbfcfe; }
.pes-preview .pp-time-head { display:flex; align-items:center; gap:6px; }
.pes-preview .pp-time-title { color:#607589; font-weight:600; }
.pes-preview .pp-time-readout { font:600 12px ui-monospace,monospace; color:#0b5f96; min-width:90px;
  text-align:center; }
.pes-preview .pp-time-state { margin-left:auto; color:#607589; font-size:12px; }
.pes-preview .pp-axis { position:relative; height:58px; margin:4px 16px 0; cursor:pointer; touch-action:none; }
.pes-preview .pp-time-bar { position:absolute; left:0; right:0; top:32px; height:14px; overflow:hidden;
  border-radius:4px; background:#e8eef4; }
.pes-preview .pp-segment { position:absolute; top:0; bottom:0; border-right:1px solid #fff;
  background:repeating-linear-gradient(135deg,#f5dcc8 0 5px,#fbede2 5px 10px); }
.pes-preview .pp-segment-current { background:repeating-linear-gradient(135deg,#cdebd8 0 5px,#e6f5ec 5px 10px);
  box-shadow:inset 0 0 0 1.5px #5fae84; }
.pes-preview .pp-event { position:absolute; bottom:12px; width:0; border-left:1.5px solid #7c5cb8; color:#6247a0; }
.pes-preview .pp-event span { position:absolute; top:-1px; left:4px; white-space:nowrap; font-size:10px;
  line-height:13px; font-weight:600; }
.pes-preview .pp-event-flip span { left:auto; right:4px; }
.pes-preview .pp-connect { border-color:#2f8f5b; color:#23744a; }
.pes-preview .pp-disconnect { border-color:#c2410c; color:#b23c0c; }
.pes-preview .pp-cursor { position:absolute; top:24px; bottom:4px; width:0; border-left:2px solid #0b5f96;
  margin-left:-1px; pointer-events:none; }
.pes-preview .pp-cursor::after { content:''; position:absolute; left:-7px; top:9px; width:12px; height:12px;
  border-radius:50%; background:#fff; border:2px solid #0b5f96; }
.pes-preview .pp-ticks { position:relative; height:18px; margin:0 16px; font-size:10px; color:#6b7f90; }
.pes-preview .pp-tick { position:absolute; transform:translateX(-50%); white-space:nowrap; }
.pes-preview .pp-scopes { padding:6px 12px; border-bottom:1px solid #e1e9f1; background:#fbfcfe; }
.pes-preview .pp-scope-panel pre { font:11px/1.45 ui-monospace,monospace; max-height:220px; overflow:auto;
  white-space:pre-wrap; margin:6px 0; padding:8px 10px; border:1px solid #e1e8ef; border-radius:5px; }
.pes-preview .pp-viewport { position:relative; overflow:auto; width:100%; background-color:#fbfdff;
  background-image:radial-gradient(#dce6ee .8px,transparent .8px); background-size:16px 16px; }
.pes-preview .pp-svg { display:block; width:100%; min-height:320px; max-height:640px; touch-action:none;
  user-select:none; }
.pes-preview .pp-edge { fill:none; stroke:#617e94; stroke-width:2; stroke-linejoin:miter; }
.pes-preview .pp-edge-inactive { stroke:#afb8be; stroke-dasharray:6 5; opacity:.25; }
.pes-preview .pp-node,.pes-preview .pp-group-action { cursor:grab; }
.pes-preview .pp-node rect { fill:#fff; stroke:#7b9aaf; stroke-width:1.7; }
.pes-preview .pp-node:hover rect { stroke:#1478ac; fill:#edf7fc; }
.pes-preview .pp-node-title { font:bold 13px system-ui,sans-serif; fill:#1c4d70; }
.pes-preview .pp-node-type { font:11px system-ui,sans-serif; fill:#60778a; }
.pes-preview .pp-port { fill:#0d72a5; }
.pes-preview .pp-port-label { fill:#42647a; font:10px system-ui,sans-serif; paint-order:stroke; stroke:#fbfdff;
  stroke-width:3px; }
.pes-preview .pp-group { fill:#e7f5f3; fill-opacity:.25; stroke:#61a799; stroke-width:1.6;
  stroke-dasharray:7 5; }
.pes-preview .pp-group-tag { fill:#e6f3ef; stroke:#61a799; }
.pes-preview .pp-group-text { fill:#25685f; font:bold 12px system-ui,sans-serif; }
.pes-preview .pp-panels { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px;
  padding:12px; border-top:1px solid #e1e9f1; }
.pes-preview .pp-panel-title { margin:0 0 8px; font-size:13px; }
.pes-preview .pp-panel details { margin:5px 0; padding:5px 8px; border:1px solid #e1e8ef;
  border-radius:5px; }
.pes-preview .pp-panel pre { font:11px/1.45 ui-monospace,monospace; max-height:260px; overflow:auto;
  white-space:pre-wrap; }
.pes-preview .pp-diagnostics { margin:0; padding:8px 12px 8px 28px; background:#fff2eb; color:#803e26; }
.pes-preview .pp-diagnostics-warning { background:#fff8e1; color:#6f5200; }
.pes-preview .pp-diagnostics button { border:0; background:none; color:inherit; padding:0 3px; text-align:left; }
.pes-preview .pp-empty { padding:28px; color:#647887; }
`;

class Renderer implements PESPreviewInstance {
  private readonly root: HTMLDivElement;
  private readonly options: PESPreviewOptions;
  private readonly markerId: string;
  private model: PESModel = parsePES('');
  private layout: PreviewLayout;
  private layouts = new Map<string, GraphLayout>();
  private viewName = '';
  private time = 0;
  private title: string;
  private readonly collapsedGroups = new Set<string>();
  private readonly openBlocks = new Set<string>();
  private openScope = '';
  private zoom = 1;
  private pan = { x: 0, y: 0 };
  private graphSvg: SVGSVGElement | null = null;
  private activeLayout: GraphLayout | null = null;
  private activeView: GraphView | null = null;
  private drag: DragState | null = null;
  private panDrag: PanState | null = null;
  private suppressClickUntil = 0;
  private destroyed = false;

  constructor(container: HTMLElement, options: PESPreviewOptions = {}) {
    if (!container || typeof container.appendChild !== 'function') {
      throw new TypeError('createPESPreview requires an HTML container');
    }
    this.options = options;
    this.title = options.title || '';
    this.layout = normalizeLayout(options.layout);
    this.markerId = `pes-preview-arrow-${++instance}`;
    this.root = element('div', 'pes-preview');
    const stylesheet = element('style');
    stylesheet.textContent = STYLES;
    this.root.appendChild(stylesheet);
    container.appendChild(this.root);
    this.render();
  }

  getModel(): PESModel { return this.model; }
  getLayout(): PreviewLayout { return JSON.parse(JSON.stringify(this.layout)) as PreviewLayout; }
  getTime(): number { return this.time; }

  update(text: string): PESModel {
    this.ensureLive();
    if (typeof text !== 'string') throw new TypeError('update(text) requires a string');
    this.model = parsePES(text);
    this.relayout();
    if (!this.model.views.some(view => view.name === this.viewName)) this.viewName = this.model.defaultView;
    this.time = this.clampTime(this.time);
    this.options.onDiagnostics?.(this.model.diagnostics);
    this.render();
    return this.model;
  }

  navigate(viewName: string): boolean {
    this.ensureLive();
    if (!this.model.views.some(view => view.name === viewName)) return false;
    this.viewName = viewName;
    this.zoom = 1;
    this.pan = { x: 0, y: 0 };
    this.render();
    return true;
  }

  setLayout(layout: unknown): void {
    this.ensureLive();
    this.layout = normalizeLayout(layout);
    this.relayout();
    this.render();
  }

  setTime(time: number): void {
    this.ensureLive();
    if (!Number.isFinite(time)) throw new TypeError('setTime requires a finite number');
    this.time = this.clampTime(time);
    this.render();
  }

  setTitle(title: string): void {
    this.ensureLive();
    this.title = String(title);
    this.render();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.root.remove();
    this.graphSvg = null;
    this.layouts.clear();
  }

  private ensureLive(): void {
    if (this.destroyed) throw new Error('PES preview instance is already destroyed');
  }

  private relayout(): void {
    this.layouts = new Map(this.model.views.map(view =>
      [view.name, layoutView(view, this.layout.views[view.name])]));
  }

  private clampTime(value: number): number {
    const first = this.model.timeline[0];
    const last = this.model.timeline[this.model.timeline.length - 1];
    return first && last ? clamp(value, first.start, last.end) : 0;
  }

  private render(): void {
    while (this.root.children.length > 1) this.root.lastElementChild?.remove();
    if (this.title) {
      const title = element('div', 'pp-title');
      title.textContent = this.title;
      title.title = this.title;
      this.root.appendChild(title);
    }
    this.renderToolbar();
    this.renderDiagnostics();
    this.renderTimeline();
    this.renderScopes();
    const view = this.model.views.find(item => item.name === this.viewName);
    if (view) this.renderGraph(view);
    else {
      const empty = element('div', 'pp-empty');
      empty.textContent = 'Add an @circuit view to preview the schematic.';
      this.root.appendChild(empty);
    }
    this.renderParameters(view);
  }

  private renderToolbar(): void {
    const toolbar = element('div', 'pp-toolbar');
    toolbar.setAttribute('role', 'toolbar');
    const strip = element('div', 'pp-strip pp-views');
    const tabs = element('div', 'pp-tabs');
    tabs.setAttribute('aria-label', 'Views');
    const scroll = (amount: number) => tabs.scrollBy?.({ left: amount, behavior: 'smooth' });
    strip.appendChild(button('‹', () => scroll(-220), 'pp-arrow'));
    for (const view of this.model.views) {
      const tab = button(view.name === 'circuit' ? 'Circuit' : view.name,
        () => this.navigate(view.name), view.name === this.viewName ? 'pp-active' : '');
      tab.setAttribute('aria-pressed', String(view.name === this.viewName));
      tabs.appendChild(tab);
    }
    strip.appendChild(tabs);
    strip.appendChild(button('›', () => scroll(220), 'pp-arrow'));
    toolbar.appendChild(strip);
    const tools = element('div', 'pp-tools');
    tools.appendChild(button('−', () => this.changeZoom(1 / 1.18)));
    tools.appendChild(button('+', () => this.changeZoom(1.18)));
    tools.appendChild(button('Fit', () => {
      this.zoom = 1; this.pan = { x: 0, y: 0 }; this.applyViewBox();
    }));
    if (Object.keys(this.layout.views[this.viewName] || {}).length) {
      tools.appendChild(button('Reset layout', () => this.resetLayout()));
    }
    toolbar.appendChild(tools);
    this.root.appendChild(toolbar);
  }

  private renderDiagnostics(): void {
    if (!this.model.diagnostics.length) return;
    const onlyWarnings = this.model.diagnostics.every(entry => entry.severity === 'warning');
    const list = element('ul', `pp-diagnostics${onlyWarnings ? ' pp-diagnostics-warning' : ''}`);
    list.setAttribute('role', 'alert');
    for (const entry of this.model.diagnostics) {
      const row = element('li');
      row.appendChild(button(`Line ${entry.range.start.line}:${entry.range.start.column} — ${entry.message}`,
        () => this.options.onSourceSelect?.(entry.range.start)));
      list.appendChild(row);
    }
    this.root.appendChild(list);
  }

  private renderTimeline(): void {
    if (!this.model.timeline.length) return;
    const first = this.model.timeline[0].start;
    const end = this.model.timeline[this.model.timeline.length - 1].end;
    const span = Math.max(end - first, Number.EPSILON);
    const position = (value: number) => `${(value - first) / span * 100}%`;
    const timeline = element('section', 'pp-timeline');
    const head = element('div', 'pp-time-head');
    const label = element('span', 'pp-time-title'); label.textContent = 'Event timeline'; head.appendChild(label);
    head.appendChild(button('‹', () => this.stepEvent(-1)));
    const readout = element('span', 'pp-time-readout'); readout.textContent = `t = ${formatTime(this.time)} s`;
    head.appendChild(readout); head.appendChild(button('›', () => this.stepEvent(1)));
    const state = element('span', 'pp-time-state');
    const current = this.model.timeline[segmentAt(this.model.timeline, this.time)];
    state.textContent = current.disconnected.length ? `Disconnected: ${current.disconnected.join(', ')}` :
      'All connections active';
    head.appendChild(state); timeline.appendChild(head);
    const axis = element('div', 'pp-axis');
    axis.setAttribute('role', 'slider'); axis.setAttribute('tabindex', '0');
    axis.setAttribute('aria-label', 'Event timeline'); axis.setAttribute('aria-valuemin', String(first));
    axis.setAttribute('aria-valuemax', String(end)); axis.setAttribute('aria-valuenow', String(this.time));
    const bar = element('div', 'pp-time-bar');
    const currentIndex = segmentAt(this.model.timeline, this.time);
    for (const segment of this.model.timeline) {
      const block = element('div', `pp-segment${segment.index === currentIndex ? ' pp-segment-current' : ''}`);
      block.setAttribute('style', `left:${position(segment.start)};width:${(segment.end - segment.start) / span * 100}%`);
      bar.appendChild(block);
    }
    axis.appendChild(bar);
    for (const event of this.model.events) {
      const mark = element('div', `pp-event pp-${event.type}${event.time > first + span * .75 ? ' pp-event-flip' : ''}`);
      mark.setAttribute('style', `left:${position(event.time)}`);
      const text = element('span'); text.textContent = `${event.type} ${event.target}`; mark.appendChild(text);
      axis.appendChild(mark);
    }
    const cursor = element('div', 'pp-cursor'); cursor.setAttribute('style', `left:${position(this.time)}`);
    axis.appendChild(cursor);
    const setFromPointer = (event: PointerEvent) => {
      const bounds = axis.getBoundingClientRect();
      this.changeTime(first + clamp((event.clientX - bounds.left) / Math.max(bounds.width, 1), 0, 1) * span);
    };
    axis.addEventListener('pointerdown', event => { axis.setPointerCapture(event.pointerId); setFromPointer(event); });
    axis.addEventListener('pointermove', event => { if (event.buttons) setFromPointer(event); });
    axis.addEventListener('keydown', event => {
      const step = span / 100;
      if (event.key === 'ArrowLeft') this.changeTime(this.time - step);
      else if (event.key === 'ArrowRight') this.changeTime(this.time + step);
      else if (event.key === 'PageUp') this.stepEvent(-1);
      else if (event.key === 'PageDown') this.stepEvent(1);
      else if (event.key === 'Home') this.changeTime(first);
      else if (event.key === 'End') this.changeTime(end);
      else return;
      event.preventDefault();
    });
    timeline.appendChild(axis);
    const ticks = element('div', 'pp-ticks');
    for (const value of niceTicks(first, end)) {
      const tick = element('span', 'pp-tick'); tick.setAttribute('style', `left:${position(value)}`);
      tick.textContent = formatTime(value); ticks.appendChild(tick);
    }
    timeline.appendChild(ticks); this.root.appendChild(timeline);
  }

  private changeTime(value: number): void {
    const next = this.clampTime(value);
    if (Math.abs(next - this.time) <= 1e-12) return;
    this.time = next; this.options.onTimeChange?.(next); this.render();
  }

  private stepEvent(direction: -1 | 1): void {
    const values = [...new Set([this.model.timeline[0]?.start, ...this.model.events.map(event => event.time),
      this.model.timeline[this.model.timeline.length - 1]?.end]
      .filter((value): value is number => typeof value === 'number'))]
      .sort((a, b) => a - b);
    const next = direction > 0 ? values.find(value => value > this.time + 1e-12) :
      [...values].reverse().find(value => value < this.time - 1e-12);
    if (next !== undefined) this.changeTime(next);
  }

  private renderScopes(): void {
    if (!this.model.globals.length) return;
    const scopes = element('section', 'pp-scopes');
    const strip = element('div', 'pp-strip');
    const tabs = element('div', 'pp-tabs'); tabs.setAttribute('aria-label', 'Settings');
    for (const block of this.model.globals) {
      const tab = button(block.name, () => { this.openScope = this.openScope === block.id ? '' : block.id; this.render(); });
      tab.setAttribute('aria-expanded', String(this.openScope === block.id)); tabs.appendChild(tab);
    }
    strip.appendChild(button('‹', () => tabs.scrollBy?.({ left: -220, behavior: 'smooth' }), 'pp-arrow'));
    strip.appendChild(tabs);
    strip.appendChild(button('›', () => tabs.scrollBy?.({ left: 220, behavior: 'smooth' }), 'pp-arrow'));
    scopes.appendChild(strip);
    const block = this.model.globals.find(item => item.id === this.openScope);
    if (block) {
      const panel = element('div', 'pp-scope-panel');
      const source = element('pre'); source.textContent = block.source; panel.appendChild(source);
      if (this.options.onSourceSelect) panel.appendChild(button(`Source: ${block.range.start.line}`,
        () => this.options.onSourceSelect?.(block.range.start)));
      scopes.appendChild(panel);
    }
    this.root.appendChild(scopes);
  }

  private renderGraph(view: GraphView): void {
    this.activeView = view;
    this.activeLayout = this.layouts.get(view.name)!;
    const viewport = element('div', 'pp-viewport');
    const picture = svg('svg', { class: 'pp-svg', role: 'img', 'aria-label': `${view.name} schematic`,
      preserveAspectRatio: 'xMidYMin meet' });
    this.graphSvg = picture;
    const definitions = svg('defs');
    const arrow = svg('marker', { id: this.markerId, viewBox: '0 0 10 10', refX: '9', refY: '5',
      markerWidth: '6', markerHeight: '6', orient: 'auto-start-reverse' });
    arrow.appendChild(svg('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: '#617e94' }));
    definitions.appendChild(arrow); picture.appendChild(definitions);
    this.applyViewBox(); this.drawScene(view);
    picture.addEventListener('pointerdown', event => {
      if (event.target !== picture) return;
      this.panDrag = { pointer: event.pointerId, x: event.clientX, y: event.clientY,
        panX: this.pan.x, panY: this.pan.y };
      picture.setPointerCapture(event.pointerId);
    });
    picture.addEventListener('pointermove', event => this.movePointer(event));
    picture.addEventListener('pointerup', event => this.endPointer(event.pointerId));
    picture.addEventListener('pointercancel', event => this.endPointer(event.pointerId));
    viewport.appendChild(picture); this.root.appendChild(viewport);
  }

  private drawScene(view: GraphView): void {
    const picture = this.graphSvg, layout = this.layouts.get(view.name);
    if (!picture || !layout) return;
    picture.querySelector?.('.pp-scene')?.remove();
    const scene = svg('g', { class: 'pp-scene' });
    const groupsById = new Map(view.groups.map(group => [group.id, group]));
    const collapsed = view.groups.filter(group => this.collapsedGroups.has(`${view.name}:${group.id}`) &&
      !view.groups.some(parent => parent !== group &&
        this.collapsedGroups.has(`${view.name}:${parent.id}`) &&
        containsGroup(parent, group.id, groupsById)));
    const collapsedOwner = new Map<string, GraphGroup>();
    const contains = (group: GraphGroup, node: string): boolean => group.members.some(member =>
      member === node || (member.startsWith('&') && groupsById.has(member) &&
        containsGroup(groupsById.get(member)!, node, groupsById)));
    for (const node of view.nodes) {
      const owner = collapsed.find(group => contains(group, node.id));
      if (owner) collapsedOwner.set(node.id, owner);
    }
    const visibleGroups = view.groups.filter(group => !view.groups.some(parent => parent !== group &&
      this.collapsedGroups.has(`${view.name}:${parent.id}`) &&
      containsGroup(parent, group.id, groupsById)));
    for (const group of [...visibleGroups].sort((a, b) =>
      (layout.groups.get(b.id)?.depth || 0) - (layout.groups.get(a.id)?.depth || 0))) {
      const bounds = layout.groups.get(group.id);
      if (!bounds) continue;
      const folded = this.collapsedGroups.has(`${view.name}:${group.id}`);
      const box = folded ? { x: bounds.x + bounds.width / 2 - 65,
        y: bounds.y + bounds.height / 2 - 22, width: 130, height: 44 } : bounds;
      const item = svg('g', { class: 'pp-group-action', role: 'button', tabindex: '0',
        'aria-label': `${folded ? 'Expand' : 'Collapse'} subsystem ${group.name}` });
      item.appendChild(svg('rect', { class: folded ? 'pp-group-tag' : 'pp-group', x: String(box.x),
        y: String(box.y), width: String(box.width), height: String(box.height), rx: '10' }));
      const label = svg('text', { class: 'pp-group-text', x: String(box.x + 9),
        y: String(box.y + (folded ? 27 : 19)) });
      label.textContent = `${folded ? '⊞' : '⊟'} ${group.name}`; item.appendChild(label);
      item.addEventListener('pointerdown', event => this.beginDrag(event, this.groupNodes(group, groupsById)));
      item.addEventListener('click', () => {
        if (Date.now() < this.suppressClickUntil) return;
        const key = `${view.name}:${group.id}`;
        if (folded) this.collapsedGroups.delete(key); else this.collapsedGroups.add(key);
        this.render();
      });
      scene.appendChild(item);
    }
    const disconnected = new Set(this.model.timeline[segmentAt(this.model.timeline, this.time)]?.disconnected || []);
    const ports = new Map<string, { anchor: Anchor; label: string }>();
    for (const edge of view.edges) {
      const from = this.endpoint(edge.from, collapsedOwner, layout);
      const to = this.endpoint(edge.to, collapsedOwner, layout);
      if (!from || !to || (from.key === to.key && from.group)) continue;
      const sides = chooseSides(this.box(from.point), this.box(to.point),
        edge.directed || !!edge.fromPort || !!edge.toPort);
      const a = this.anchor(from.point, sides[0], edge.fromPort, view);
      const b = this.anchor(to.point, sides[1], edge.toPort, view);
      const inactive = disconnected.has(edge.from) || disconnected.has(edge.to) || disconnected.has(edge.id);
      const line = svg('path', { class: `pp-edge${inactive ? ' pp-edge-inactive' : ''}`,
        d: pathData(routeEdge(a, b)) });
      if (edge.directed && !inactive) line.setAttribute('marker-end', `url(#${this.markerId})`);
      scene.appendChild(line);
      if (!from.group && edge.fromPort) ports.set(`${edge.from}:${edge.fromPort}`, { anchor: a, label: edge.fromPort });
      if (!to.group && edge.toPort) ports.set(`${edge.to}:${edge.toPort}`, { anchor: b, label: edge.toPort });
    }
    for (const node of view.nodes) {
      if (collapsedOwner.has(node.id)) continue;
      const point = layout.nodes.get(node.id)!;
      const item = svg('g', { class: 'pp-node', role: 'button', tabindex: '0',
        'aria-label': `${node.name}${node.type ? ' (' + node.type + ')' : ''}` });
      item.appendChild(svg('rect', { x: String(point.x - point.width / 2), y: String(point.y - point.height / 2),
        width: String(point.width), height: String(point.height), rx: '8' }));
      const label = svg('text', { class: 'pp-node-title', x: String(point.x),
        y: String(point.y - (node.type ? 5 : -4)), 'text-anchor': 'middle' });
      label.textContent = node.name; item.appendChild(label);
      if (node.type) {
        const type = svg('text', { class: 'pp-node-type', x: String(point.x), y: String(point.y + 15),
          'text-anchor': 'middle' });
        type.textContent = node.controllerType ? `${node.type}, ${node.controllerType}` : node.type;
        item.appendChild(type);
      }
      item.addEventListener('pointerdown', event => this.beginDrag(event, [node.id]));
      item.addEventListener('click', () => {
        if (Date.now() < this.suppressClickUntil) return;
        if (!this.navigate(node.name)) this.options.onSourceSelect?.(node.source.start);
      });
      scene.appendChild(item);
    }
    for (const { anchor, label } of ports.values()) this.drawPort(scene, anchor, label);
    picture.appendChild(scene);
  }

  private groupNodes(group: GraphGroup, groups: Map<string, GraphGroup>, seen = new Set<string>()): string[] {
    if (seen.has(group.id)) return [];
    seen.add(group.id);
    return group.members.flatMap(member => member.startsWith('&') && groups.has(member) ?
      this.groupNodes(groups.get(member)!, groups, seen) : [member]);
  }

  private beginDrag(event: PointerEvent, ids: string[]): void {
    event.stopPropagation();
    if (!this.activeView || !this.graphSvg) return;
    const offsets = this.layout.views[this.activeView.name] || {};
    const bounds = this.graphSvg.getBoundingClientRect();
    this.drag = { pointer: event.pointerId, startX: event.clientX, startY: event.clientY, ids,
      base: new Map(ids.map(id => [id, { ...(offsets[id] || { dx: 0, dy: 0 }) }])),
      positions: new Map([...this.activeLayout!.nodes].map(([id, point]) =>
        [id, { x: point.x, y: point.y }])),
      scale: this.activeLayout!.width / Math.max(bounds.width, 1) / this.zoom, moved: false };
    this.graphSvg.setPointerCapture(event.pointerId);
  }

  private movePointer(event: PointerEvent): void {
    if (this.drag && event.pointerId === this.drag.pointer && this.activeView && this.activeLayout) {
      let dx = (event.clientX - this.drag.startX) * this.drag.scale;
      let dy = (event.clientY - this.drag.startY) * this.drag.scale;
      if (!this.drag.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
      this.drag.moved = true;
      dx = Math.round(dx / GRID) * GRID; dy = Math.round(dy / GRID) * GRID;
      [dx, dy] = this.alignDelta(this.drag, dx, dy);
      const offsets = { ...(this.layout.views[this.activeView.name] || {}) };
      for (const id of this.drag.ids) {
        const base = this.drag.base.get(id)!;
        const next = { dx: base.dx + dx, dy: base.dy + dy };
        if (next.dx || next.dy) offsets[id] = next; else delete offsets[id];
      }
      if (Object.keys(offsets).length) this.layout.views[this.activeView.name] = offsets;
      else delete this.layout.views[this.activeView.name];
      const layout = layoutView(this.activeView, offsets);
      this.layouts.set(this.activeView.name, layout); this.activeLayout = layout;
      this.applyViewBox(); this.drawScene(this.activeView);
      return;
    }
    if (this.panDrag && event.pointerId === this.panDrag.pointer && this.activeLayout) {
      const bounds = this.graphSvg!.getBoundingClientRect();
      const scale = this.activeLayout.width / Math.max(bounds.width, 1) / this.zoom;
      this.pan = { x: this.panDrag.panX - (event.clientX - this.panDrag.x) * scale,
        y: this.panDrag.panY - (event.clientY - this.panDrag.y) * scale };
      this.applyViewBox();
    }
  }

  private alignDelta(drag: DragState, dx: number, dy: number): [number, number] {
    const moving = new Set(drag.ids), first = drag.positions.get(drag.ids[0]);
    if (!first) return [dx, dy];
    for (const [id, point] of drag.positions) {
      if (moving.has(id)) continue;
      if (Math.abs(first.x + dx - point.x) <= ALIGN_DISTANCE) dx = point.x - first.x;
      if (Math.abs(first.y + dy - point.y) <= ALIGN_DISTANCE) dy = point.y - first.y;
    }
    return [dx, dy];
  }

  private endPointer(pointer: number): void {
    if (this.drag?.pointer === pointer) {
      const moved = this.drag.moved; this.drag = null;
      if (moved) {
        this.suppressClickUntil = Date.now() + 100;
        this.options.onLayoutChange?.(this.getLayout()); this.render();
      }
    }
    if (this.panDrag?.pointer === pointer) this.panDrag = null;
  }

  private resetLayout(): void {
    if (!this.layout.views[this.viewName]) return;
    delete this.layout.views[this.viewName]; this.relayout();
    this.options.onLayoutChange?.(this.getLayout()); this.render();
  }

  private endpoint(id: string, collapsed: Map<string, GraphGroup>, layout: GraphLayout): Endpoint | null {
    const group = collapsed.get(id);
    if (group) {
      const bounds = layout.groups.get(group.id);
      return bounds ? { key: group.id, group: true, point: { id, x: bounds.x + bounds.width / 2,
        y: bounds.y + bounds.height / 2, width: 130, height: 44 } } : null;
    }
    const point = layout.nodes.get(id);
    return point ? { key: id, group: false, point } : null;
  }

  private box(point: LayoutNode): Box {
    return { x: point.x, y: point.y, width: point.width, height: point.height };
  }

  private anchor(point: LayoutNode, side: Side, port: string | undefined, view: GraphView): Anchor {
    const node = view.nodes.find(item => item.id === point.id);
    const ports = node?.ports || [], index = port ? ports.indexOf(port) : -1;
    const along = index < 0 ? 0 : ((index + 1) / (ports.length + 1) - .5);
    let x = point.x, y = point.y;
    if (side === 'left' || side === 'right') {
      x += (side === 'right' ? 1 : -1) * point.width / 2; y += along * (point.height - 12);
    } else {
      y += (side === 'bottom' ? 1 : -1) * point.height / 2; x += along * (point.width - 12);
    }
    return { x, y, side, box: this.box(point) };
  }

  private drawPort(parent: SVGGElement, anchor: Anchor, label: string): void {
    parent.appendChild(svg('circle', { class: 'pp-port', cx: String(anchor.x), cy: String(anchor.y), r: '3' }));
    const text = svg('text', { class: 'pp-port-label', x: String(anchor.x + 5), y: String(anchor.y - 6) });
    text.textContent = label; parent.appendChild(text);
  }

  private applyViewBox(): void {
    if (!this.graphSvg || !this.activeLayout) return;
    const { left, top, width, height } = this.activeLayout;
    this.graphSvg.setAttribute('viewBox', `${left + this.pan.x} ${top + this.pan.y} ${width / this.zoom} ${height / this.zoom}`);
  }

  private changeZoom(factor: number): void { this.zoom = clamp(this.zoom * factor, .35, 3.5); this.applyViewBox(); }

  private renderParameters(view: GraphView | undefined): void {
    if (!view?.parameters.length) return;
    const panels = element('div', 'pp-panels'), panel = element('section');
    const heading = element('h3', 'pp-panel-title'); heading.textContent = 'Parameters'; panel.appendChild(heading);
    for (const block of view.parameters) {
      const details = element('details'); details.open = this.openBlocks.has(block.id);
      const summary = element('summary'); summary.textContent = `${block.name}: ${block.type}`; details.appendChild(summary);
      const source = element('pre'); source.textContent = block.source; details.appendChild(source);
      details.addEventListener('toggle', () => details.open ? this.openBlocks.add(block.id) : this.openBlocks.delete(block.id));
      panel.appendChild(details);
    }
    panels.appendChild(panel); this.root.appendChild(panels);
  }
}

function containsGroup(root: GraphGroup, target: string, groups: Map<string, GraphGroup>,
                       seen = new Set<string>()): boolean {
  if (seen.has(root.id)) return false;
  seen.add(root.id);
  return root.members.some(member => member === target ||
    (groups.has(member) && containsGroup(groups.get(member)!, target, groups, seen)));
}

export function createPESPreview(container: HTMLElement,
                                 options?: PESPreviewOptions): PESPreviewInstance {
  return new Renderer(container, options);
}
