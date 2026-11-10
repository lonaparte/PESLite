import test from 'node:test';
import assert from 'node:assert/strict';
import { createPESPreview, parsePES, layoutView } from '../test-output/index.js';

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

@base:
  s_base: 2.0e6

@simulation:
  t_end: 1.0
  solver: {dt: 25e-6}

@events:
  connect_vsc: {type: connect, target: VSC, t: 0.2}
  disconnect_vsc: {type: disconnect, target: VSC, t: 0.7}
`;

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
  const text = `@circuit\nA -- Unit\n@Unit\nX.out -> Y.in\nUnit: Converter, custom\n  X: PID\n    kp: 1.0\n  Y: Output\n@simulation:\n  t_end: 1\n`;
  const parsed = parsePES(text);
  assert.deepEqual(parsed.diagnostics, []);
  assert.equal(parsed.views[0].nodes.find(node => node.name === 'Unit').controllerType, 'custom');
  assert.equal(parsed.views[1].nodes.find(node => node.name === 'X').type, 'PID');
  assert.deepEqual(parsed.views[1].nodes.find(node => node.name === 'X').ports, ['out']);
});

test('reports syntax and YAML errors with precise line numbers and does not mutate source', () => {
  const original = '@circuit\nA --\n&Loop: &Loop\n@simulation:\n  t_end: {bad\n';
  const model = parsePES(original);
  assert.ok(model.diagnostics.some(diag => diag.range.start.line === 2 && diag.severity === 'error'));
  assert.ok(model.diagnostics.some(diag => diag.range.start.line === 3 && diag.message.includes('cyclic')));
  assert.ok(model.diagnostics.some(diag => diag.range.start.line === 5 && diag.message.includes('unclosed')));
  assert.equal(original, '@circuit\nA --\n&Loop: &Loop\n@simulation:\n  t_end: {bad\n');
});

test('accepts event block mappings, same-topology intervals and separate ports', () => {
  const model = parsePES(`@circuit\nS.a -> N.b\n@simulation:\n  t_end: 3\n@events:\n  a:\n    type: disconnect\n    target: S\n    t: 1\n  b:\n    type: disconnect\n    target: S\n    t: 2\n`);
  assert.deepEqual(model.diagnostics, []);
  assert.deepEqual(model.timeline.map(segment => [segment.start, segment.end]), [[0, 1], [1, 2], [2, 3]]);
  assert.deepEqual(model.timeline[1].disconnected, model.timeline[2].disconnected);
  assert.equal(model.views[0].edges[0].fromPort, 'a');
  assert.equal(model.views[0].edges[0].toPort, 'b');
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
  getBoundingClientRect() { return { width: 900, height: 440 }; }
  setPointerCapture() { }
  click() {
    for (const handler of this.listeners.get('click') || []) handler({ target: this, currentTarget: this });
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
    const tab = findAll(host, node => node.tagName === 'button' && node.textContent === '0.2–0.7 s')[0];
    assert.ok(tab);
    tab.click();
    assert.equal(findAll(host, node => (node.getAttribute('class') || '').includes('pp-edge-inactive')).length, 0);
    const toggle = findAll(host, node => node.getAttribute('aria-label') === 'Collapse subsystem Loads')[0];
    toggle.click();
    assert.ok(findAll(host, node => node.getAttribute('aria-label') === 'Expand subsystem Loads').length);
    const vsc = findAll(host, node => node.getAttribute('aria-label') === 'VSC (vsc)')[0];
    vsc.click();
    assert.ok(findAll(host, node => node.getAttribute('aria-label') === 'PLL (SRFPLL)').length);
    assert.ok(findAll(host, node => node.tagName === 'button' && node.textContent === '← Back').length);
    preview.update('@circuit\nA ->');
    assert.ok(findAll(host, node => node.getAttribute('role') === 'alert').length);
    preview.destroy();
    assert.equal(host.children.length, 0);
    assert.throws(() => preview.update(SAMPLE), /destroyed/);
  } finally { globalThis.document = old; }
});


test('accepts a host-normalized legacy YAML graph without requiring a Python parser in the bundle', () => {
  const old = globalThis.document;
  globalThis.document = {
    createElement: tag => new FakeElement(tag),
    createElementNS: (_namespace, tag) => new FakeElement(tag),
  };
  try {
    const host = new FakeElement('div');
    const preview = createPESPreview(host);
    const graph = parsePES('@circuit\nGrid -- VSC\n');
    graph.views[0].nodes.find(node => node.id === 'VSC').name = 'controller';
    graph.views[0].nodes.find(node => node.id === 'VSC').type = 'vsc';
    preview.updateModel(graph);
    assert.equal(preview.getModel(), graph);
    assert.ok(findAll(host, node => node.getAttribute('aria-label') === 'controller (vsc)').length);
    assert.throws(() => preview.updateModel({}), /requires a PESModel/);
    preview.destroy();
  } finally { globalThis.document = old; }
});

test('exposes optional host-neutral edge selections without modifying source', () => {
  const old = globalThis.document;
  globalThis.document = {
    createElement: tag => new FakeElement(tag),
    createElementNS: (_namespace, tag) => new FakeElement(tag),
  };
  try {
    const host = new FakeElement('div');
    const selected = [];
    const preview = createPESPreview(host, {
      onEdgeSelect: (edge, view) => selected.push([edge.id, view]),
    });
    preview.update('@circuit\nA.out -> B.in\n');
    const edge = findAll(host, node => node.tagName === 'path' &&
      node.getAttribute('aria-label')?.startsWith('Select connection '))[0];
    assert.ok(edge);
    edge.click();
    assert.equal(selected.length, 1);
    assert.equal(selected[0][1], 'circuit');
    assert.equal(selected[0][0], preview.getModel().views[0].edges[0].id);
    preview.destroy();
  } finally { globalThis.document = old; }
});
