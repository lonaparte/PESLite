import { parsePES } from './parser.js';
import { layoutView } from './layout.js';
import type {
  Diagnostic, GraphGroup, GraphLayout, GraphView,
  LayoutNode, PESModel, SourceLocation,
} from './types.js';

const NS = 'http://www.w3.org/2000/svg';
let instance = 0;

export interface PESPreviewOptions {
  /** Called when the user selects a declaration or diagnostic. */
  onSourceSelect?: (location: SourceLocation) => void;
  onDiagnostics?: (diagnostics: readonly Diagnostic[]) => void;
}
export interface PESPreviewInstance {
  update(text: string): PESModel;
  navigate(viewName: string): boolean;
  getModel(): PESModel;
  destroy(): void;
}
function element<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (className) el.className = className;
  return el;
}
function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string> = {}): SVGElementTagNameMap[K] {
  const el = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  return el;
}
function button(label: string, action: () => void, className?: string): HTMLButtonElement {
  const result = element('button', className);
  result.type = 'button';
  result.textContent = label;
  result.addEventListener('click', action);
  return result;
}
function formatTime(value: number): string { return Number(value.toPrecision(5)).toString(); }
function targetMatches(target: string, name: string): boolean {
  return target === name || target === `units.${name}` || target === `sources.${name}` ||
    target === `elements.${name}` || target === `buses.${name}`;
}

const STYLES = `
.pes-preview { font: 13px/1.5 system-ui, sans-serif; color: #24384a; background: #fff;
  border: 1px solid #d5e0eb; border-radius: 9px; overflow: hidden; }
.pes-preview * { box-sizing: border-box; }
.pes-preview button { font: inherit; cursor: pointer; color: inherit; border: 1px solid #cbd8e4;
  background: #fff; border-radius: 5px; padding: 4px 9px; }
.pes-preview button:hover { border-color: #4d90be; }
.pes-preview button:focus-visible, .pes-preview summary:focus-visible { outline: 2px solid #287aaa; }
.pes-preview .pp-toolbar, .pes-preview .pp-timeline { display: flex; gap: 7px; flex-wrap: wrap;
  align-items: center; padding: 8px 12px; border-bottom: 1px solid #e1e9f1; }
.pes-preview .pp-toolbar { background: #f5f9fc; }
.pes-preview .pp-title { font-weight: 700; color: #174564; margin-right: auto; }
.pes-preview .pp-breadcrumb { font-weight: 600; color: #265f88; }
.pes-preview .pp-timeline { background: #fbfcfe; }
.pes-preview .pp-timeline-label { color: #607589; margin-right: 8px; }
.pes-preview .pp-active { color: #065e98 !important; border-color: #76aed0 !important;
  background: #e4f2fb !important; }
.pes-preview .pp-viewport { overflow: auto; width: 100%; background-color: #fbfdff;
  background-image: radial-gradient(#dce6ee 0.8px, transparent 0.8px);
  background-size: 16px 16px; }
.pes-preview .pp-svg { display: block; width: 100%; min-height: 320px; max-height: 640px;
  touch-action: none; user-select: none; }
.pes-preview .pp-edge { fill: none; stroke: #617e94; stroke-width: 2; }
.pes-preview .pp-edge-inactive { stroke: #afb8be; stroke-dasharray: 6 5; opacity: .25; }
.pes-preview .pp-node { cursor: pointer; }
.pes-preview .pp-node rect { fill: #fff; stroke: #7b9aaf; stroke-width: 1.7; }
.pes-preview .pp-node:hover rect { stroke: #1478ac; fill: #edf7fc; }
.pes-preview .pp-node-title { font: bold 13px system-ui, sans-serif; fill: #1c4d70; }
.pes-preview .pp-node-type { font: 11px system-ui, sans-serif; fill: #60778a; }
.pes-preview .pp-port { fill: #0d72a5; }
.pes-preview .pp-port-label { fill: #42647a; font: 10px system-ui, sans-serif; }
.pes-preview .pp-group { fill: #e7f5f3; fill-opacity: .25; stroke: #61a799;
  stroke-width: 1.6; stroke-dasharray: 7 5; }
.pes-preview .pp-group-tag { fill: #e6f3ef; stroke: #61a799; stroke-width: 1; }
.pes-preview .pp-group-text { fill: #25685f; font: bold 12px system-ui, sans-serif; }
.pes-preview .pp-group-action { cursor: pointer; }
.pes-preview .pp-panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px; padding: 12px; border-top: 1px solid #e1e9f1; }
.pes-preview .pp-panel-title { margin: 0 0 8px; font-size: 13px; font-weight: 700; }
.pes-preview .pp-panel details { margin: 5px 0; padding: 5px 8px; border: 1px solid #e1e8ef;
  border-radius: 5px; background: #fff; }
.pes-preview .pp-panel summary { cursor: pointer; overflow-wrap: anywhere; }
.pes-preview .pp-panel pre { font: 11px/1.45 ui-monospace, monospace; max-height: 260px;
  overflow: auto; white-space: pre-wrap; word-break: break-word; margin: 8px 0; }
.pes-preview .pp-diagnostics { margin: 0; padding: 8px 12px 8px 28px; background: #fff2eb;
  color: #803e26; border-bottom: 1px solid #f0c8b9; }
.pes-preview .pp-diagnostics button { border: 0; background: none; color: inherit;
  padding: 0 3px; text-align: left; }
.pes-preview .pp-empty { padding: 28px; color: #647887; }
`;

class Renderer implements PESPreviewInstance {
  private readonly root: HTMLDivElement;
  private readonly options: PESPreviewOptions;
  private readonly markerId: string;
  private model: PESModel = parsePES('');
  private layouts = new Map<string, GraphLayout>();
  private viewName = '';
  private history: string[] = [];
  private segment = 0;
  private readonly collapsedGroups = new Set<string>();
  private readonly openBlocks = new Set<string>();
  private zoom = 1;
  private pan = { x: 0, y: 0 };
  private graphSvg: SVGSVGElement | null = null;
  private activeLayout: GraphLayout | null = null;
  private destroyed = false;

  constructor(container: HTMLElement, options: PESPreviewOptions = {}) {
    if (!container || typeof container.appendChild !== 'function') {
      throw new TypeError('createPESPreview requires an HTML container');
    }
    this.options = options;
    this.markerId = `pes-preview-arrow-${++instance}`;
    this.root = element('div', 'pes-preview');
    const stylesheet = element('style');
    stylesheet.textContent = STYLES;
    this.root.appendChild(stylesheet);
    container.appendChild(this.root);
    this.render();
  }

  getModel(): PESModel { return this.model; }

  update(text: string): PESModel {
    this.ensureLive();
    if (typeof text !== 'string') throw new TypeError('update(text) requires a string');
    this.model = parsePES(text);
    this.layouts = new Map(this.model.views.map(view => [view.name, layoutView(view)]));
    if (!this.model.views.some(view => view.name === this.viewName)) {
      this.viewName = this.model.defaultView;
      this.history = [];
    }
    this.segment = Math.min(this.segment, Math.max(0, this.model.timeline.length - 1));
    this.options.onDiagnostics?.(this.model.diagnostics);
    this.render();
    return this.model;
  }

  navigate(viewName: string): boolean {
    this.ensureLive();
    if (!this.model.views.some(view => view.name === viewName)) return false;
    if (this.viewName && this.viewName !== viewName) this.history.push(this.viewName);
    this.viewName = viewName;
    this.zoom = 1;
    this.pan = { x: 0, y: 0 };
    this.render();
    return true;
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

  private render(): void {
    // Keep the scoped stylesheet while replacing the owned DOM subtree.
    while (this.root.children.length > 1) this.root.lastElementChild?.remove();
    const toolbar = element('div', 'pp-toolbar');
    const title = element('span', 'pp-title');
    title.textContent = 'PES schematic preview';
    toolbar.appendChild(title);
    if (this.history.length) toolbar.appendChild(button('← Back', () => {
      this.viewName = this.history.pop() || this.model.defaultView;
      this.zoom = 1;
      this.pan = { x: 0, y: 0 };
      this.render();
    }));
    for (const view of this.model.views) {
      const switcher = button(view.name === 'circuit' ? 'Circuit' : view.name,
        () => { this.navigate(view.name); },
        view.name === this.viewName ? 'pp-active pp-breadcrumb' : 'pp-breadcrumb');
      switcher.setAttribute('aria-pressed', String(view.name === this.viewName));
      toolbar.appendChild(switcher);
    }
    toolbar.appendChild(button('+', () => this.changeZoom(1.18)));
    toolbar.appendChild(button('−', () => this.changeZoom(1 / 1.18)));
    toolbar.appendChild(button('Fit', () => {
      this.zoom = 1; this.pan = { x: 0, y: 0 }; this.applyViewBox();
    }));
    this.root.appendChild(toolbar);
    if (this.model.diagnostics.length) {
      const list = element('ul', 'pp-diagnostics');
      list.setAttribute('role', 'alert');
      for (const entry of this.model.diagnostics) {
        const li = element('li');
        li.appendChild(button(`Line ${entry.range.start.line}:${entry.range.start.column} — ${entry.message}`,
          () => this.options.onSourceSelect?.(entry.range.start)));
        list.appendChild(li);
      }
      this.root.appendChild(list);
    }
    if (this.model.timeline.length) {
      const timeline = element('nav', 'pp-timeline');
      timeline.setAttribute('aria-label', 'Event intervals');
      const title = element('span', 'pp-timeline-label');
      title.textContent = 'Event timeline';
      timeline.appendChild(title);
      for (const interval of this.model.timeline) {
        const tab = button(`${formatTime(interval.start)}–${formatTime(interval.end)} s`, () => {
          this.segment = interval.index;
          this.render();
        }, interval.index === this.segment ? 'pp-active' : '');
        tab.setAttribute('aria-pressed', String(interval.index === this.segment));
        tab.title = `Connections after events at t = ${formatTime(interval.start)} s`;
        timeline.appendChild(tab);
      }
      this.root.appendChild(timeline);
    }
    const view = this.model.views.find(item => item.name === this.viewName);
    if (!view) {
      const empty = element('div', 'pp-empty');
      empty.textContent = 'Add an @circuit section to preview your schematic.';
      this.root.appendChild(empty);
    } else {
      this.renderGraph(view);
    }
    this.renderPanels(view);
  }

  private renderGraph(view: GraphView): void {
    const layout = this.layouts.get(view.name)!;
    this.activeLayout = layout;
    const viewport = element('div', 'pp-viewport');
    const picture = svg('svg', {
      class: 'pp-svg', role: 'img', 'aria-label': `${view.name} schematic`,
      preserveAspectRatio: 'xMidYMin meet',
    });
    this.graphSvg = picture;
    this.applyViewBox();
    const definitions = svg('defs');
    const arrow = svg('marker', { id: this.markerId, viewBox: '0 0 10 10',
      refX: '9', refY: '5', markerWidth: '6', markerHeight: '6', orient: 'auto-start-reverse' });
    arrow.appendChild(svg('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: '#617e94' }));
    definitions.appendChild(arrow);
    picture.appendChild(definitions);

    const groupsById = new Map(view.groups.map(group => [group.id, group]));
    function contains(group: GraphGroup, node: string, visited = new Set<string>()): boolean {
      if (visited.has(group.id)) return false;
      visited.add(group.id);
      return group.members.some(member => member === node ||
        (member.startsWith('&') && groupsById.has(member) &&
          contains(groupsById.get(member)!, node, visited)));
    }
    const collapsed = view.groups.filter(group => this.collapsedGroups.has(`${view.name}:${group.id}`)
      && !view.groups.some(parent => parent.id !== group.id &&
        this.collapsedGroups.has(`${view.name}:${parent.id}`) &&
        parent.members.some(member => member === group.id ||
          (member.startsWith('&') && groupsById.has(member) &&
            containsGroup(groupsById.get(member)!, group.id, groupsById)))));
    const collapsedOwner = new Map<string, GraphGroup>();
    for (const node of view.nodes) {
      const owner = collapsed.find(group => contains(group, node.id));
      if (owner) collapsedOwner.set(node.id, owner);
    }
    const visibleGroups = view.groups.filter(group => !view.groups.some(parent =>
      parent.id !== group.id && this.collapsedGroups.has(`${view.name}:${parent.id}`) &&
      containsGroup(parent, group.id, groupsById)));
    for (const group of [...visibleGroups].sort((a, b) =>
      (layout.groups.get(b.id)?.depth || 0) - (layout.groups.get(a.id)?.depth || 0))) {
      const bounds = layout.groups.get(group.id);
      if (!bounds) continue;
      const folded = this.collapsedGroups.has(`${view.name}:${group.id}`);
      const g = svg('g', { class: 'pp-group-action', role: 'button', tabindex: '0',
        'aria-label': `${folded ? 'Expand' : 'Collapse'} subsystem ${group.name}` });
      const box = folded ? { x: bounds.x + bounds.width / 2 - 65,
        y: bounds.y + bounds.height / 2 - 22, width: 130, height: 44 } : bounds;
      g.appendChild(svg('rect', { class: folded ? 'pp-group-tag' : 'pp-group',
        x: String(box.x), y: String(box.y), width: String(box.width),
        height: String(box.height), rx: '10' }));
      const text = svg('text', { class: 'pp-group-text', x: String(box.x + 9),
        y: String(box.y + (folded ? 27 : 19)) });
      text.textContent = `${folded ? '⊞' : '⊟'} ${group.name}`;
      g.appendChild(text);
      const toggle = () => {
        const key = `${view.name}:${group.id}`;
        if (folded) this.collapsedGroups.delete(key);
        else this.collapsedGroups.add(key);
        this.render();
      };
      g.addEventListener('click', toggle);
      g.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggle(); }
      });
      picture.appendChild(g);
    }
    const disconnected = new Set(this.model.timeline[this.segment]?.disconnected || []);
    for (const edge of view.edges) {
      const from = this.endpoint(edge.from, collapsedOwner, layout);
      const to = this.endpoint(edge.to, collapsedOwner, layout);
      if (!from || !to || (from.key === to.key && from.group)) continue;
      const a = this.anchor(from, to.point, edge.fromPort, view);
      const b = this.anchor(to, from.point, edge.toPort, view);
      const inactive = [...disconnected].some(target =>
        targetMatches(target, edge.from) || targetMatches(target, edge.to) ||
        target === edge.id || target === `${edge.from}--${edge.to}` ||
        target === `${edge.from}->${edge.to}`);
      const line = svg('path', { class: `pp-edge${inactive ? ' pp-edge-inactive' : ''}`,
        d: `M ${a.x} ${a.y} L ${b.x} ${b.y}` });
      if (edge.directed && !inactive) line.setAttribute('marker-end', `url(#${this.markerId})`);
      const title = svg('title');
      title.textContent = `${edge.from}${edge.fromPort ? '.' + edge.fromPort : ''} ${edge.directed ? '→' : '—'} ${edge.to}${edge.toPort ? '.' + edge.toPort : ''}${inactive ? ' (disconnected)' : ''}`;
      line.appendChild(title);
      picture.appendChild(line);
      if (!from.group && edge.fromPort) this.drawPort(picture, a, edge.fromPort);
      if (!to.group && edge.toPort) this.drawPort(picture, b, edge.toPort);
    }
    for (const node of view.nodes) {
      if (collapsedOwner.has(node.id)) continue;
      const point = layout.nodes.get(node.id)!;
      const group = svg('g', { class: 'pp-node', role: 'button', tabindex: '0',
        'aria-label': `${node.name}${node.type ? ' (' + node.type + ')' : ''}` });
      group.appendChild(svg('rect', { x: String(point.x - point.width / 2),
        y: String(point.y - point.height / 2),
        width: String(point.width), height: String(point.height), rx: '8' }));
      const label = svg('text', { class: 'pp-node-title', x: String(point.x),
        y: String(point.y - (node.type ? 5 : -4)), 'text-anchor': 'middle' });
      label.textContent = node.name;
      group.appendChild(label);
      if (node.type) {
        const type = svg('text', { class: 'pp-node-type', x: String(point.x),
          y: String(point.y + 15), 'text-anchor': 'middle' });
        type.textContent = node.controllerType ? `${node.type}, ${node.controllerType}` : node.type;
        group.appendChild(type);
      }
      const action = () => {
        if (this.model.views.some(view => view.name === node.name) && node.name !== this.viewName) {
          this.navigate(node.name);
        } else this.options.onSourceSelect?.(node.source.start);
      };
      group.addEventListener('click', action);
      group.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); action(); }
      });
      picture.appendChild(group);
    }
    let dragging: { pointer: number; x: number; y: number; panX: number; panY: number } | null = null;
    picture.addEventListener('pointerdown', event => {
      if (event.target !== picture) return;
      dragging = { pointer: event.pointerId, x: event.clientX, y: event.clientY,
        panX: this.pan.x, panY: this.pan.y };
      picture.setPointerCapture(event.pointerId);
    });
    picture.addEventListener('pointermove', event => {
      if (!dragging || event.pointerId !== dragging.pointer) return;
      const rect = picture.getBoundingClientRect();
      const scale = layout.width / (Math.max(rect.width, 1) * this.zoom);
      this.pan = { x: dragging.panX - (event.clientX - dragging.x) * scale,
        y: dragging.panY - (event.clientY - dragging.y) * scale };
      this.applyViewBox();
    });
    picture.addEventListener('pointerup', () => { dragging = null; });
    picture.addEventListener('pointercancel', () => { dragging = null; });
    picture.addEventListener('wheel', event => {
      event.preventDefault();
      this.changeZoom(event.deltaY < 0 ? 1.1 : 1 / 1.1);
    }, { passive: false });
    viewport.appendChild(picture);
    this.root.appendChild(viewport);
  }

  private endpoint(id: string, collapsed: Map<string, GraphGroup>, layout: GraphLayout) {
    const group = collapsed.get(id);
    if (group) {
      const bounds = layout.groups.get(group.id);
      if (!bounds) return null;
      return { key: group.id, group: true, point: { id, x: bounds.x + bounds.width / 2,
        y: bounds.y + bounds.height / 2, width: 130, height: 44 } };
    }
    const point = layout.nodes.get(id);
    return point ? { key: id, group: false, point } : null;
  }

  private anchor(endpoint: { point: LayoutNode; group: boolean }, opposite: LayoutNode,
                 port: string | undefined, view: GraphView): { x: number; y: number } {
    const point = endpoint.point;
    const right = opposite.x >= point.x;
    const vertically = Math.abs(opposite.y - point.y) > Math.abs(opposite.x - point.x) * 1.5;
    if (vertically && !port) return { x: point.x, y: point.y +
      (opposite.y >= point.y ? point.height / 2 : -point.height / 2) };
    const node = view.nodes.find(item => item.id === point.id);
    const ports = node?.ports || [];
    const index = port ? ports.indexOf(port) : -1;
    const y = index < 0 ? point.y : point.y +
      ((index + 1) / (ports.length + 1) - 0.5) * (point.height - 12);
    return { x: point.x + (right ? point.width / 2 : -point.width / 2), y };
  }

  private drawPort(picture: SVGSVGElement, pos: { x: number; y: number }, label: string): void {
    picture.appendChild(svg('circle', { class: 'pp-port', cx: String(pos.x),
      cy: String(pos.y), r: '3' }));
    const text = svg('text', { class: 'pp-port-label', x: String(pos.x + 5),
      y: String(pos.y - 6) });
    text.textContent = label;
    picture.appendChild(text);
  }

  private applyViewBox(): void {
    if (!this.graphSvg || !this.activeLayout) return;
    const { width, height } = this.activeLayout;
    this.graphSvg.setAttribute('viewBox', `${this.pan.x} ${this.pan.y} ${width / this.zoom} ${height / this.zoom}`);
  }

  private changeZoom(factor: number): void {
    this.zoom = Math.min(3.5, Math.max(0.35, this.zoom * factor));
    this.applyViewBox();
  }

  private renderPanels(view: GraphView | undefined): void {
    const panels = element('div', 'pp-panels');
    for (const [label, blocks] of [
      ['Parameters', view?.parameters || []],
      ['Global settings', this.model.globals],
    ] as const) {
      if (!blocks.length) continue;
      const panel = element('section', 'pp-panel');
      const heading = element('h3', 'pp-panel-title');
      heading.textContent = label;
      panel.appendChild(heading);
      for (const block of blocks) {
        const details = element('details');
        const key = `${view?.name || ''}:${block.id}`;
        details.open = this.openBlocks.has(key);
        const summary = element('summary');
        summary.textContent = `${block.name}${block.type !== 'global' ? ': ' + block.type : ''}`;
        details.appendChild(summary);
        const source = element('pre');
        source.textContent = block.source;
        details.appendChild(source);
        if (this.options.onSourceSelect) details.appendChild(button(
          `Source: ${block.range.start.line}`, () => this.options.onSourceSelect?.(block.range.start)));
        details.addEventListener('toggle', () => {
          if (details.open) this.openBlocks.add(key);
          else this.openBlocks.delete(key);
        });
        panel.appendChild(details);
      }
      panels.appendChild(panel);
    }
    if (panels.childElementCount) this.root.appendChild(panels);
  }
}

function containsGroup(root: GraphGroup, target: string, groups: Map<string, GraphGroup>,
                       visited = new Set<string>()): boolean {
  if (visited.has(root.id)) return false;
  visited.add(root.id);
  return root.members.some(member => member === target ||
    (groups.has(member) && containsGroup(groups.get(member)!, target, groups, visited)));
}

/** Instantiate the same browser-only renderer in any local Web or VS Code host. */
export function createPESPreview(container: HTMLElement,
                                 options?: PESPreviewOptions): PESPreviewInstance {
  return new Renderer(container, options);
}
