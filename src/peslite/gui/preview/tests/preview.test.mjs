import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildTimeline, createPESPreview, parsePES, layoutView, normalizeLayout,
  routeEdge, pathData, chooseSides, niceTicks, segmentAt,
} from '../test-output/index.js';

const SAMPLE = `# Standalone PES preview syntax
@circuit
Grid -- PCC -- VSC
PCC -- Load1
PCC -- Load2
&Loads: Load1, Load2
&System: &Loads, Grid, PCC, VSC

@VSC
PLL.theta -> Current.theta
DC.id_ref -> Current.id_ref
Current -> PWM

VSC: vsc, gfl
  ac_filter:
    l_f_pu: 0.2
  PLL: SRFPLL
    kp_pu: 20
    ki_pu: 1200
  Current: CurrentLoop
    bandwidth: 200
  PWM: ThreePhaseModulation
  DC: DCVoltageControl
    kp: 0.8

base:
  s_base: 2.0e6

simulation:
  t_end: 1.0
  solver: {dt: 25e-6}

events:
  connect_vsc: {type: connect, target: VSC, t: 0.2}
  disconnect_vsc: {type: disconnect, target: VSC, t: 0.7}
`;

test('parses every bundled native simulation file with matching event targets', () => {
  const examples = fileURLToPath(new URL('../../../../../examples/', import.meta.url));
  const files = readdirSync(examples).filter(name => name.endsWith('.pes'))
    .map(name => join(examples, name));
  files.push(join(examples, 'templates', 'template.pes'));
  for (const file of files) {
    const model = parsePES(readFileSync(file, 'utf8'));
    assert.deepEqual(model.diagnostics.filter(item => item.severity === 'error'), [], file);
    const circuit = model.views.find(view => view.name === 'circuit');
    const nodes = new Set(circuit?.nodes.map(node => node.name));
    for (const event of model.events) assert.ok(nodes.has(event.target), `${file}: ${event.target}`);
  }
});

test('parses unit-scoped loops, nested group membership, named ports and YAML blocks', () => {
  const model = parsePES(SAMPLE);
  assert.deepEqual(model.diagnostics, []);
  assert.equal(model.defaultView, 'circuit');
  assert.deepEqual(model.views.map(view => view.name), ['circuit', 'VSC']);
  const circuit = model.views[0];
  const controller = model.views[1];
  assert.deepEqual(circuit.nodes.map(node => node.name), ['Grid', 'PCC', 'VSC', 'Load1', 'Load2']);
  assert.equal(circuit.nodes.find(node => node.name === 'VSC').type, 'vsc');
  assert.equal(circuit.nodes.find(node => node.name === 'VSC').controllerType, 'gfl');
  assert.equal(circuit.groups.length, 2);
  assert.equal(circuit.groups.find(group => group.id === '&Loads').parent, '&System');
  assert.deepEqual(controller.nodes.map(node => node.name), ['PLL', 'Current', 'DC', 'PWM']);
  assert.equal(controller.edges[0].fromPort, 'theta');
  assert.equal(controller.edges[0].toPort, 'theta');
  assert.equal(controller.edges[1].fromPort, 'id_ref');
  assert.equal(controller.edges[1].toPort, 'id_ref');
  const pll = controller.parameters.find(block => block.name === 'PLL');
  assert.ok(pll?.source.includes('kp_pu: 20'));
  assert.ok(pll?.source.includes('ki_pu: 1200'));
  assert.ok(controller.parameters.some(block => block.name === 'VSC' && block.children.includes(pll.id)));
  assert.ok(model.globals.some(block => block.name === 'simulation' && block.source.includes('t_end: 1.0')));
});

test('creates separate timeline segments without recalculating layout', () => {
  const model = parsePES(SAMPLE);
  assert.deepEqual(model.timeline.map(segment => [segment.start, segment.end]),
    [[0, 0.2], [0.2, 0.7], [0.7, 1]]);
  assert.ok(model.timeline[0].disconnected.includes('VSC'));
  assert.ok(!model.timeline[1].disconnected.includes('VSC'));
  assert.ok(model.timeline[2].disconnected.includes('VSC'));
  const a = layoutView(model.views[0]);
  const b = layoutView(model.views[0]);
  assert.deepEqual([...a.nodes], [...b.nodes]);
  assert.deepEqual([...a.groups], [...b.groups]);
  assert.ok(a.nodes.get('PCC').x < a.nodes.get('VSC').x);
  assert.notEqual(a.nodes.get('Load1').y, a.nodes.get('Load2').y);
});

test('allows forward references and nested control views with independent names', () => {
  const text = `@circuit\nA -- Unit\n@Unit\nX.out -> Y.in\nUnit: Converter, custom # displayed types\n  X: PID\n    kp: 1.0\n  Y: Output\nsimulation:\n  t_end: 1\n`;
  const parsed = parsePES(text);
  assert.deepEqual(parsed.diagnostics, []);
  assert.equal(parsed.views[0].nodes.find(node => node.name === 'Unit').controllerType, 'custom');
  assert.equal(parsed.views[1].nodes.find(node => node.name === 'X').type, 'PID');
  assert.deepEqual(parsed.views[1].nodes.find(node => node.name === 'X').ports, ['out']);
});

test('parses parameter domains without @ or a fixed domain-name list', () => {
  const parsed = parsePES(`base: {s_base: 1e6}\n@circuit\nGrid -- Unit\n@Unit\nLoop -> PWM\nUnit: Converter\n  Loop: CurrentControl\n  PWM: DutyModulation\ncustom_parameters:\n  enabled: true\n  matrix: [[1, 2],\n           [3, 4]]\nsimulation:\n  t_end: 2\n`);
  assert.deepEqual(parsed.diagnostics, []);
  assert.deepEqual(parsed.views.map(view => view.name), ['circuit', 'Unit']);
  assert.deepEqual(parsed.globals.map(block => block.name),
    ['base', 'custom_parameters', 'simulation']);
  assert.ok(parsed.globals.find(block => block.name === 'custom_parameters')?.source
    .includes('enabled: true'));
  assert.deepEqual(parsed.timeline.map(segment => [segment.start, segment.end]), [[0, 2]]);
});

test('reserves @ headers for rendered views', () => {
  const parsed = parsePES('@base:\n  s_base: 1e6\n@circuit\nA -- B\n');
  assert.ok(parsed.diagnostics.some(diag => diag.message === 'invalid @view header'));
  assert.ok(!parsed.globals.some(block => block.name === 'base'));
});

test('identifies nested control declarations from graph structure, not capitalization', () => {
  const parsed = parsePES(`@circuit\nGrid -- Unit\n@Unit\nLoop -> PWM\nUnit: Converter\n  mode: PWM\n  Loop: CurrentControl\n  PWM: DutyModulation\n`);
  assert.deepEqual(parsed.diagnostics, []);
  const view = parsed.views[1];
  assert.deepEqual(view.nodes.map(node => node.name), ['Loop', 'PWM']);
  assert.equal(view.nodes.find(node => node.name === 'Loop').type, 'CurrentControl');
  assert.ok(!view.parameters.some(block => block.name === 'mode'));
});

test('reports syntax and YAML errors with precise line numbers and does not mutate source', () => {
  const original = '@circuit\nA --\n&Loop: &Loop\nsimulation:\n  t_end: {bad\n';
  const model = parsePES(original);
  assert.ok(model.diagnostics.some(diag => diag.range.start.line === 2 && diag.severity === 'error'));
  assert.ok(model.diagnostics.some(diag => diag.range.start.line === 3 && diag.message.includes('cyclic')));
  assert.ok(model.diagnostics.some(diag => diag.range.start.line === 5 && diag.message.includes('unclosed')));
  assert.equal(original, '@circuit\nA --\n&Loop: &Loop\nsimulation:\n  t_end: {bad\n');
});

test('accepts event block mappings, same-topology intervals and separate ports', () => {
  const model = parsePES(`@circuit\nS.a -> N.b\nsimulation:\n  t_end: 3\nevents:\n  a:\n    type: disconnect\n    target: S\n    t: 1\n  b:\n    type: disconnect\n    target: S\n    t: 2\n`);
  assert.deepEqual(model.diagnostics, []);
  assert.deepEqual(model.timeline.map(segment => [segment.start, segment.end]), [[0, 1], [1, 2], [2, 3]]);
  assert.deepEqual(model.timeline[1].disconnected, model.timeline[2].disconnected);
  assert.equal(model.views[0].edges[0].fromPort, 'a');
  assert.equal(model.views[0].edges[0].toPort, 'b');
});

test('preserves event source lines independently of indentation width', () => {
  const model = parsePES(`@circuit\nA -- B\nevents:\n    bad:\n      type: disconnect\n      target: A\n      t: invalid\n`);
  const error = model.diagnostics.find(diag => diag.message.includes('invalid connect/disconnect'));
  assert.equal(error?.range.start.line, 4);
});

test('builds event timelines in one ordered state sweep', () => {
  const events = Array.from({ length: 2000 }, (_, index) => ({
    name: `event_${index}`,
    type: index % 2 ? 'disconnect' : 'connect',
    target: `node_${index % 20}`,
    time: (1999 - index) / 1000,
    line: index + 1,
  }));
  const timeline = parsePES(`@circuit\nA -- B\nsimulation:\n  t_end: 2\n`).timeline;
  assert.deepEqual(timeline.map(segment => [segment.start, segment.end]), [[0, 2]]);
  const first = [...events].sort((a, b) => a.time - b.time || a.line - b.line)[0];
  assert.equal(first.time, 0);
  const segments = buildTimeline(events, 2);
  assert.equal(segments.length, 2000);
  assert.equal(segments.at(-1).end, 2);
});

class FakeElement {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.listeners = new Map();
    this.attributes = new Map();
    this.className = '';
    this.textContent = '';
    this.open = false;
    this.dataset = {};
    this.style = {};
  }
  get lastElementChild() { return this.children.at(-1) || null; }
  get childElementCount() { return this.children.length; }
  appendChild(child) { this.children.push(child); child.parent = this; return child; }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this);
  }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name) || null; }
  addEventListener(name, handler) {
    this.listeners.set(name, [...(this.listeners.get(name) || []), handler]);
  }
  getBoundingClientRect() { return { left: 0, top: 0, width: 900, height: 440 }; }
  setPointerCapture() { }
  scrollBy() { }
  querySelector(selector) {
    return findAll(this, node => selector.startsWith('.') &&
      String(node.className).split(' ').includes(selector.slice(1)))[0];
  }
  click() {
    for (const handler of this.listeners.get('click') || []) handler({ target: this, currentTarget: this });
  }
  fire(name, event = {}) {
    for (const handler of this.listeners.get(name) || []) handler({ target: this, currentTarget: this,
      button: 0, buttons: 1, pointerId: 1, clientX: 0, clientY: 0,
      preventDefault() {}, stopPropagation() {}, ...event });
  }
}
function findAll(node, predicate) {
  return (predicate(node) ? [node] : []).concat(...node.children.map(child => findAll(child, predicate)));
}

test('renders host-neutral SVG and updates timeline/folding/navigation without source IO', () => {
  const old = globalThis.document;
  globalThis.document = {
    createElement: tag => new FakeElement(tag),
    createElementNS: (_namespace, tag) => new FakeElement(tag),
  };
  try {
    const host = new FakeElement('div');
    const preview = createPESPreview(host);
    preview.update(SAMPLE);
    const firstX = findAll(host, node => node.tagName === 'g' &&
      node.getAttribute('aria-label') === 'VSC (vsc)')[0];
    assert.ok(firstX);
    assert.ok(findAll(host, node => (node.getAttribute('class') || '').includes('pp-edge-inactive')).length);
    const slider = findAll(host, node => node.getAttribute('role') === 'slider')[0];
    assert.ok(slider);
    slider.fire('keydown', { key: 'PageDown' });
    assert.equal(findAll(host, node => (node.getAttribute('class') || '').includes('pp-edge-inactive')).length, 0);
    const toggle = findAll(host, node => node.getAttribute('aria-label') === 'Collapse subsystem Loads')[0];
    toggle.click();
    assert.ok(findAll(host, node => node.getAttribute('aria-label') === 'Expand subsystem Loads').length);
    const vsc = findAll(host, node => node.getAttribute('aria-label') === 'VSC (vsc)')[0];
    vsc.click();
    assert.ok(findAll(host, node => node.getAttribute('aria-label') === 'PLL (SRFPLL)').length);
    assert.equal(findAll(host, node => node.tagName === 'button' && /Back/.test(node.textContent)).length, 0);
    preview.update('@circuit\nA ->');
    assert.ok(findAll(host, node => node.getAttribute('role') === 'alert').length);
    preview.destroy();
    assert.equal(host.children.length, 0);
    assert.throws(() => preview.update(SAMPLE), /destroyed/);
  } finally { globalThis.document = old; }
});

test('assigns named declarations to their views and exposes unconnected views', () => {
  const model = parsePES(`@circuit
Grid -- VSC
@VSC
PLL -> PWM
@Spare
A -> B
VSC: vsc, gfl
  PLL: SRFPLL
  PWM: ThreePhaseModulation
`);
  const circuit = model.views[0], vsc = model.views[1], spare = model.views[2];
  assert.deepEqual(vsc.nodes.map(node => [node.name, node.type]),
    [['PLL', 'SRFPLL'], ['PWM', 'ThreePhaseModulation']]);
  assert.deepEqual(spare.nodes.map(node => node.name), ['A', 'B']);
  assert.equal(circuit.nodes.find(node => node.name === 'VSC').type, 'vsc');
  assert.ok(circuit.nodes.some(node => node.name === 'Spare'));
  assert.ok(model.diagnostics.some(entry => entry.severity === 'warning' && /Spare/.test(entry.message)));
});

test('applies and validates manual layout offsets', () => {
  const view = parsePES(SAMPLE).views[0];
  const automatic = layoutView(view);
  const moved = layoutView(view, { VSC: { dx: -600, dy: 32 } });
  assert.equal(moved.nodes.get('VSC').x, automatic.nodes.get('VSC').x - 600);
  assert.equal(moved.nodes.get('VSC').y, automatic.nodes.get('VSC').y + 32);
  assert.ok(moved.left < 0);
  assert.deepEqual(normalizeLayout({ views: { circuit: {
    VSC: { dx: 8, dy: -4 }, bad: { dx: 'x', dy: 1 }, zero: { dx: 0, dy: 0 },
  } } }).views, { circuit: { VSC: { dx: 8, dy: -4 } } });
});

test('routes edges orthogonally with square corners', () => {
  const a = { x: 100, y: 100, width: 142, height: 64 };
  const b = { x: 400, y: 220, width: 142, height: 64 };
  assert.deepEqual(chooseSides(a, b, false), ['right', 'left']);
  const points = routeEdge({ x: 171, y: 100, side: 'right', box: a },
    { x: 329, y: 220, side: 'left', box: b });
  assert.ok(points.slice(1).every((point, index) =>
    point.x === points[index].x || point.y === points[index].y));
  assert.ok(!pathData(points).includes('Q'));
});

test('selects event intervals and round time ticks', () => {
  const timeline = parsePES(SAMPLE).timeline;
  assert.equal(segmentAt(timeline, 0.2), 1);
  assert.equal(segmentAt(timeline, 0.7), 2);
  assert.deepEqual(niceTicks(0, 1, 5), [0, 0.2, 0.4, 0.6, 0.8, 1]);
});

test('drags blocks without changing source text and reports cacheable offsets', () => {
  const old = globalThis.document;
  globalThis.document = {
    createElement: tag => new FakeElement(tag),
    createElementNS: (_namespace, tag) => new FakeElement(tag),
  };
  try {
    const host = new FakeElement('div'), layouts = [];
    const preview = createPESPreview(host, { onLayoutChange: layout => layouts.push(layout) });
    preview.update(SAMPLE);
    const picture = findAll(host, node => node.tagName === 'svg')[0];
    const vsc = findAll(host, node => node.getAttribute('aria-label') === 'VSC (vsc)')[0];
    vsc.fire('pointerdown', { clientX: 100, clientY: 100 });
    picture.fire('pointermove', { clientX: 180, clientY: 140 });
    picture.fire('pointerup');
    assert.equal(layouts.length, 1);
    assert.ok(layouts[0].views.circuit.VSC.dx || layouts[0].views.circuit.VSC.dy);
    assert.equal(preview.getModel().views[0].source.start.line, 2);
    findAll(host, node => node.tagName === 'button' && node.textContent === 'Fit')[0].click();
    assert.ok(preview.getLayout().views.circuit.VSC);
    preview.destroy();
  } finally { globalThis.document = old; }
});

test('accepts host-normalized models without replacing renderer features', () => {
  const old = globalThis.document;
  globalThis.document = {
    createElement: tag => new FakeElement(tag),
    createElementNS: (_namespace, tag) => new FakeElement(tag),
  };
  try {
    const host = new FakeElement('div');
    const preview = createPESPreview(host);
    const model = parsePES('@circuit\nGrid -- VSC\n');
    const vsc = model.views[0].nodes.find(node => node.id === 'VSC');
    vsc.name = 'controller';
    vsc.type = 'vsc';
    assert.equal(preview.updateModel(model), model);
    assert.equal(preview.getModel(), model);
    assert.ok(findAll(host, node => node.getAttribute('aria-label') === 'controller (vsc)').length);
    assert.throws(() => preview.updateModel({}), /requires a PESModel/);
    preview.destroy();
  } finally { globalThis.document = old; }
});

test('reports node and edge selections through the shared renderer API', () => {
  const old = globalThis.document;
  globalThis.document = {
    createElement: tag => new FakeElement(tag),
    createElementNS: (_namespace, tag) => new FakeElement(tag),
  };
  try {
    const host = new FakeElement('div'), nodes = [], edges = [];
    const preview = createPESPreview(host, {
      onNodeSelect: (node, view) => nodes.push([node.id, view]),
      onEdgeSelect: (edge, view) => edges.push([edge.id, view]),
    });
    preview.update('@circuit\nA.out -> B.in\n');
    findAll(host, node => node.getAttribute('aria-label') === 'A')[0].click();
    findAll(host, node => node.getAttribute('aria-label')?.startsWith('Select connection '))[0].click();
    assert.deepEqual(nodes, [['A', 'circuit']]);
    assert.deepEqual(edges, [[preview.getModel().views[0].edges[0].id, 'circuit']]);
    preview.destroy();
  } finally { globalThis.document = old; }
});
