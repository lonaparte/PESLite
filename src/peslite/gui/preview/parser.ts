import type {
  Diagnostic, EventSegment, GraphEdge, GraphEvent, GraphGroup, GraphNode, GraphView,
  ParameterBlock, PESModel, SourceLocation, SourceRange,
} from './types.js';

const REFERENCE = /^([A-Za-z_][\w-]*)(?:\.([A-Za-z_][\w-]*))?$/;
const HEADER = /^@([A-Za-z_][\w-]*)(:?)\s*$/;
const GLOBALS = new Set(['base', 'simulation', 'events', 'meta', 'signals']);

type Value = null | string | number | boolean | Value[] | { [key: string]: Value };
type Mapping = { [key: string]: Value };
interface ParsedMapping {
  value: Mapping;
  keyLines: Map<Mapping, Map<string, number>>;
}
interface Section { name: string; global: boolean; line: number; end: number }
interface SourceLine { text: string; line: number }

function loc(line: number, column = 1): SourceLocation { return { line, column }; }
function range(line: number, start = 1, end = start + 1): SourceRange {
  return { start: loc(line, start), end: loc(line, end) };
}
function diagnostic(output: Diagnostic[], line: number, column: number, message: string,
                    severity: 'error' | 'warning' = 'error'): void {
  output.push({ severity, message, range: range(line, column) });
}
function indent(text: string): number { return text.match(/^ */)![0].length; }

/** Strip only comments outside quotes and YAML flow collections. */
function stripComment(input: string): string {
  let quote = '', escaped = false;
  for (let i = 0; i < input.length; i++) {
    const ch = input[i];
    if (quote) {
      if (quote === '"' && ch === '\\' && !escaped) { escaped = true; continue; }
      if (ch === quote && !escaped) {
        if (quote === "'" && input[i + 1] === "'") { i++; continue; }
        quote = '';
      }
      escaped = false;
    } else if (ch === "'" || ch === '"') quote = ch;
    else if (ch === '#' && (i === 0 || /\s/.test(input[i - 1]))) return input.slice(0, i);
  }
  return input;
}

/** Split a flow collection on commas not inside quotes or nested brackets. */
function splitTopLevel(text: string, separator = ','): string[] {
  const results: string[] = [];
  let depth = 0, quote = '', escaped = false, start = 0;
  for (let index = 0; index < text.length; index++) {
    const ch = text[index];
    if (quote) {
      if (quote === '"' && ch === '\\' && !escaped) { escaped = true; continue; }
      if (ch === quote && !escaped) {
        if (quote === "'" && text[index + 1] === "'") { index++; continue; }
        quote = '';
      }
      escaped = false;
    } else if (ch === '"' || ch === "'") quote = ch;
    else if (ch === '[' || ch === '{') depth++;
    else if (ch === ']' || ch === '}') depth--;
    else if (ch === separator && depth === 0) {
      results.push(text.slice(start, index).trim()); start = index + 1;
    }
  }
  if (quote || depth !== 0) throw new Error('unclosed quote or flow collection');
  results.push(text.slice(start).trim());
  return results;
}

function colonOutsideFlow(text: string): number {
  let quote = '', escaped = false, depth = 0;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quote) {
      if (quote === '"' && ch === '\\' && !escaped) { escaped = true; continue; }
      if (ch === quote && !escaped) {
        if (quote === "'" && text[i + 1] === "'") { i++; continue; }
        quote = '';
      }
      escaped = false;
    } else if (ch === '"' || ch === "'") quote = ch;
    else if (ch === '[' || ch === '{') depth++;
    else if (ch === ']' || ch === '}') depth--;
    else if (ch === ':' && depth === 0) return i;
  }
  return -1;
}

function yamlValue(raw: string): Value {
  const text = raw.trim();
  if (text === '' || text === '~' || text === 'null') return null;
  if (text === 'true') return true;
  if (text === 'false') return false;
  if (/^[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?$/i.test(text)) return Number(text);
  if (text.startsWith('"')) {
    if (!text.endsWith('"')) throw new Error('unclosed double-quoted value');
    try { return JSON.parse(text) as string; }
    catch { throw new Error('invalid double-quoted value'); }
  }
  if (text.startsWith("'")) {
    if (!text.endsWith("'")) throw new Error('unclosed single-quoted value');
    return text.slice(1, -1).replace(/''/g, "'");
  }
  if (text.startsWith('{')) {
    if (!text.endsWith('}')) throw new Error('unclosed inline mapping');
    const map: Mapping = {};
    if (text.slice(1, -1).trim()) {
      for (const pair of splitTopLevel(text.slice(1, -1))) {
        const at = colonOutsideFlow(pair);
        if (at < 1) throw new Error(`invalid inline mapping entry: ${pair}`);
        const key = String(yamlValue(pair.slice(0, at)));
        if (Object.prototype.hasOwnProperty.call(map, key)) throw new Error(`duplicate YAML key: ${key}`);
        map[key] = yamlValue(pair.slice(at + 1));
      }
    }
    return map;
  }
  if (text.startsWith('[')) {
    if (!text.endsWith(']')) throw new Error('unclosed inline sequence');
    return text.slice(1, -1).trim() ? splitTopLevel(text.slice(1, -1)).map(yamlValue) : [];
  }
  // Opaque YAML scalars are retained as strings. This is not a PESLite validator.
  return text;
}

/** Lightweight YAML *mapping* reader for global settings and event metadata. */
function parseYamlMapping(lines: SourceLine[], diagnostics: Diagnostic[]): ParsedMapping {
  const relevant = lines.filter(line => stripComment(line.text).trim() !== '');
  const keyLines = new Map<Mapping, Map<string, number>>();
  let index = 0;
  function read(level: number): Mapping {
    const result: Mapping = {};
    const locations = new Map<string, number>();
    keyLines.set(result, locations);
    while (index < relevant.length) {
      const row = relevant[index];
      const column = indent(row.text);
      if (column < level) break;
      if (column > level) {
        diagnostic(diagnostics, row.line, column + 1, 'unexpected YAML indentation');
        index++; continue;
      }
      const expression = stripComment(row.text.trimEnd()).trim();
      const at = colonOutsideFlow(expression);
      if (at < 1) {
        diagnostic(diagnostics, row.line, column + 1, 'expected a YAML key: value mapping');
        index++; continue;
      }
      const key = expression.slice(0, at).trim().replace(/^(['"])(.*)\1$/, '$2');
      if (!key || Object.prototype.hasOwnProperty.call(result, key)) {
        diagnostic(diagnostics, row.line, column + 1, `empty or duplicate YAML key: ${key}`);
        index++; continue;
      }
      locations.set(key, row.line);
      const value = expression.slice(at + 1).trim();
      index++;
      if (value === '' && index < relevant.length && indent(relevant[index].text) > column) {
        result[key] = read(indent(relevant[index].text));
      } else if (/^[|>][-+]?\s*$/.test(value)) {
        // YAML block strings are display-only; skip their indented contents.
        while (index < relevant.length && indent(relevant[index].text) > column) index++;
        result[key] = value;
      } else {
        try { result[key] = yamlValue(value); }
        catch (error) { diagnostic(diagnostics, row.line, column + at + 2, String(error)); }
      }
    }
    return result;
  }
  return { value: relevant.length ? read(indent(relevant[0].text)) : {}, keyLines };
}

/** Split a schematic path into node references and -- / -> separators. */
function connectionPath(expression: string): { names: string[]; arrows: string[] } | null {
  if (!/--|->/.test(expression)) return null;
  const pieces = expression.split(/(--|->)/).map(piece => piece.trim());
  if (pieces.length < 3 || pieces.length % 2 === 0) return { names: [], arrows: [] };
  const names = pieces.filter((_, i) => i % 2 === 0);
  const arrows = pieces.filter((_, i) => i % 2 === 1);
  return { names, arrows };
}

function parseView(section: Section, lines: string[], diagnostics: Diagnostic[]): GraphView {
  const nodes = new Map<string, GraphNode>();
  const edges: GraphEdge[] = [];
  const groups: GraphGroup[] = [];
  const parameters: ParameterBlock[] = [];
  const source = { start: loc(section.line), end: loc(section.end, (lines[section.end - 1] || '').length + 1) };
  const view: GraphView = { name: section.name, nodes: [], edges, groups, parameters, source };
  let active: { block: ParameterBlock; indent: number; start: number;
    nested?: { block: ParameterBlock; start: number } } | null = null;

  function node(name: string, line: number, column: number): GraphNode {
    let existing = nodes.get(name);
    if (!existing) {
      existing = { id: name, name, firstLine: line, source: range(line, column, column + name.length), ports: [] };
      nodes.set(name, existing);
    }
    return existing;
  }
  function definitionEnd(endLine: number): void {
    if (!active) return;
    const previous = active.block;
    if (active.nested) {
      active.nested.block.range.end = loc(endLine, (lines[endLine - 1] || '').length + 1);
      active.nested.block.source = lines.slice(active.nested.start - 1, endLine).join('\n');
    }
    previous.range.end = loc(endLine, (lines[endLine - 1] || '').length + 1);
    previous.source = lines.slice(active.start - 1, endLine).join('\n');
    active = null;
  }
  for (let lineNumber = section.line + 1; lineNumber <= section.end; lineNumber++) {
    const raw = lines[lineNumber - 1] ?? '';
    const expression = stripComment(raw).trim();
    if (!expression) continue;
    if (/\t/.test(raw.slice(0, raw.length - raw.trimStart().length))) {
      diagnostic(diagnostics, lineNumber, 1, 'tab indentation is not supported');
      continue;
    }
    const level = indent(raw);
    if (level > 0) {
      if (!active) {
        diagnostic(diagnostics, lineNumber, level + 1, 'indented YAML without a parameter declaration');
        continue;
      }
      if (level <= active.indent) {
        diagnostic(diagnostics, lineNumber, level + 1, 'invalid parameter block indentation');
      }
      // YAML parameters are opaque to the numerical backend but their local
      // syntax still needs to be safe to display and source-located.
      const field = expression.match(/^([A-Za-z_][\w.-]*)\s*:\s*(.*)$/);
      if (!field && !expression.startsWith('- ')) {
        diagnostic(diagnostics, lineNumber, level + 1, 'expected an indented YAML key: value');
      }
      if (field && /^[{["']/.test(field[2])) {
        try { yamlValue(field[2]); }
        catch (error) { diagnostic(diagnostics, lineNumber, level + 1, String(error)); }
      }
      const candidate = expression.match(/^([A-Za-z_][\w-]*)\s*:\s*([A-Za-z_][\w.-]*)\s*$/);
      const nested = candidate && nodes.has(candidate[1]) ? candidate : null;
      if (level === active.indent + 2) {
        if (active.nested) {
          active.nested.block.range.end = loc(lineNumber - 1, (lines[lineNumber - 2] || '').length + 1);
          active.nested.block.source = lines.slice(active.nested.start - 1, lineNumber - 1).join('\n');
          active.nested = undefined;
        }
      }
      if (nested && level === active.indent + 2 && active.block.name === section.name) {
        if (parameters.some(item => item.id === `${section.name}.${active!.block.name}.${nested[1]}`)) {
          diagnostic(diagnostics, lineNumber, level + 1, `duplicate loop declaration: ${nested[1]}`);
          continue;
        }
        const item = node(nested[1], lineNumber, level + 1);
        item.type = nested[2];
        const id = `${section.name}.${active.block.name}.${nested[1]}`;
        const nestedBlock: ParameterBlock = {
          id, name: nested[1], type: nested[2],
          source: raw, range: range(lineNumber, level + 1, raw.length + 1),
          parent: active.block.id, children: [],
        };
        active.block.children.push(id);
        parameters.push(nestedBlock);
        active.nested = { block: nestedBlock, start: lineNumber };
        item.parameterBlock = id;
      }
      continue;
    }
    definitionEnd(lineNumber - 1);
    if (expression.startsWith('&')) {
      const match = expression.match(/^&([A-Za-z_][\w-]*)\s*:\s*(.*)$/);
      if (!match) {
        diagnostic(diagnostics, lineNumber, 1, 'expected &Group: Member, Member');
        continue;
      }
      const members = match[2].split(',').map(value => value.trim());
      if (members.length === 0 || members.some(name =>
        !/^&?[A-Za-z_][\w-]*$/.test(name))) {
        diagnostic(diagnostics, lineNumber, 1, 'invalid subsystem member list');
        continue;
      }
      if (groups.some(group => group.name === match[1])) {
        diagnostic(diagnostics, lineNumber, 1, `duplicate subsystem: ${match[1]}`);
        continue;
      }
      groups.push({ id: '&' + match[1], name: match[1], members,
        line: lineNumber, source: range(lineNumber, 1, raw.length + 1) });
      for (const member of members) if (!member.startsWith('&')) node(member, lineNumber, raw.indexOf(member) + 1);
      continue;
    }
    const chain = connectionPath(expression);
    if (chain) {
      if (!chain.arrows.length || chain.names.some(name => !REFERENCE.test(name))) {
        diagnostic(diagnostics, lineNumber, 1, 'invalid connection; expected A.port -- B.port or A -> B');
        continue;
      }
      const endpoints = chain.names.map(name => {
        const matched = name.match(REFERENCE)!;
        const component = node(matched[1], lineNumber, Math.max(1, raw.indexOf(matched[1]) + 1));
        const port = matched[2];
        if (port && !component.ports.includes(port)) component.ports.push(port);
        return { id: matched[1], port };
      });
      for (let i = 0; i < chain.arrows.length; i++) {
        edges.push({ id: `${section.name}:${lineNumber}:${i}`, from: endpoints[i].id,
          to: endpoints[i + 1].id, fromPort: endpoints[i].port, toPort: endpoints[i + 1].port,
          directed: chain.arrows[i] === '->', line: lineNumber,
          source: range(lineNumber, 1, raw.length + 1) });
      }
      continue;
    }
    const definition = expression.match(/^([A-Za-z_][\w-]*)\s*:\s*([A-Za-z_][\w.-]*(?:\s*,\s*[A-Za-z_][\w.-]*)?)\s*$/);
    if (definition) {
      const id = `${section.name}.${definition[1]}`;
      if (parameters.some(item => item.id === id)) {
        diagnostic(diagnostics, lineNumber, 1, `duplicate component declaration: ${definition[1]}`);
        continue;
      }
      const typeNames = definition[2].split(',').map(value => value.trim());
      const block: ParameterBlock = {
        id, name: definition[1], type: typeNames.join(', '), source: raw,
        range: range(lineNumber, 1, raw.length + 1), children: [],
      };
      parameters.push(block);
      active = { block, indent: level, start: lineNumber };
      if (definition[1] !== section.name) {
        const component = node(definition[1], lineNumber, 1);
        component.type = typeNames[0];
        if (typeNames.length > 1) component.controllerType = typeNames[1];
        component.parameterBlock = id;
      }
      continue;
    }
    diagnostic(diagnostics, lineNumber, 1, 'unrecognized schematic statement');
  }
  definitionEnd(section.end);
  const groupByName = new Map(groups.map(group => [group.id, group]));
  for (const group of groups) {
    for (const member of group.members) {
      if (!member.startsWith('&')) continue;
      const child = groupByName.get(member);
      if (!child) { diagnostic(diagnostics, group.line, 1, `unknown subsystem: ${member}`); continue; }
      if (child.parent && child.parent !== group.id) {
        diagnostic(diagnostics, group.line, 1, `subsystem ${member} has multiple parents`);
      } else child.parent = group.id;
    }
  }
  function checkCycles(group: GraphGroup, visited: Set<string>): void {
    if (visited.has(group.id)) {
      diagnostic(diagnostics, group.line, 1, `cyclic subsystem membership: ${group.id}`);
      return;
    }
    const next = new Set(visited).add(group.id);
    for (const member of group.members) {
      const nested = groupByName.get(member);
      if (nested) checkCycles(nested, next);
    }
  }
  for (const group of groups) checkCycles(group, new Set());
  view.nodes = [...nodes.values()].sort((a, b) => a.firstLine - b.firstLine);
  return view;
}

function readEvents(parsed: ParsedMapping | undefined, diagnostics: Diagnostic[],
                    startLine: number): GraphEvent[] {
  const raw = parsed?.value;
  if (!raw) return [];
  const result: GraphEvent[] = [];
  for (const [name, data] of Object.entries(raw)) {
    const event = data && typeof data === 'object' && !Array.isArray(data) ? data as Mapping : null;
    const line = parsed.keyLines.get(raw)?.get(name) || startLine;
    if (!event) { diagnostic(diagnostics, line, 1, `event ${name} must be a mapping`); continue; }
    if (event.type !== 'connect' && event.type !== 'disconnect') continue;
    if (typeof event.target !== 'string' || typeof event.t !== 'number' ||
        !Number.isFinite(event.t) || event.t < 0) {
      diagnostic(diagnostics, line, 1, `invalid connect/disconnect event: ${name}`);
      continue;
    }
    result.push({ name, type: event.type, target: event.target, time: event.t, line });
  }
  return result;
}

/** Connection/node state at a fixed segment; no layout is recomputed. */
export function disconnectedAt(events: readonly GraphEvent[], start: number): string[] {
  const ordered = orderEvents(events);
  const inactive = initialDisconnected(ordered);
  for (const event of ordered) {
    if (event.time > start + 1e-12) break;
    if (event.type === 'connect') inactive.delete(event.target);
    else inactive.add(event.target);
  }
  return [...inactive].sort();
}

function orderEvents(events: readonly GraphEvent[]): GraphEvent[] {
  return [...events].sort((a, b) => a.time - b.time || a.line - b.line);
}

function initialDisconnected(events: readonly GraphEvent[]): Set<string> {
  const first = new Map<string, GraphEvent['type']>();
  for (const event of events) if (!first.has(event.target)) first.set(event.target, event.type);
  return new Set([...first].filter(([, type]) => type === 'connect').map(([target]) => target));
}

export function buildTimeline(events: readonly GraphEvent[], endTime: number, initial = 0): EventSegment[] {
  const end = Number.isFinite(endTime) && endTime > initial ? endTime :
    Math.max(initial + 1, ...events.map(event => event.time + 1));
  const boundaries = [...new Set([initial, end, ...events.map(event => event.time)
    .filter(time => time > initial && time < end)])].sort((a, b) => a - b);
  const ordered = orderEvents(events);
  const inactive = initialDisconnected(ordered);
  const result: EventSegment[] = [];
  let eventIndex = 0;
  for (let index = 0; index < boundaries.length - 1; index++) {
    const start = boundaries[index];
    while (eventIndex < ordered.length && ordered[eventIndex].time <= start + 1e-12) {
      const event = ordered[eventIndex++];
      if (event.type === 'connect') inactive.delete(event.target);
      else inactive.add(event.target);
    }
    result.push({ index, start, end: boundaries[index + 1], disconnected: [...inactive].sort() });
  }
  return result;
}

/** Parse the standalone @-section PES preview syntax without running PESLite/Python. */
export function parsePES(text: string): PESModel {
  const lines = text.replace(/^\uFEFF/, '').split(/\r\n|\n|\r/);
  const diagnostics: Diagnostic[] = [];
  const sections: Section[] = [];
  let current: Section | null = null;
  for (let index = 0; index < lines.length; index++) {
    const raw = stripComment(lines[index]);
    if (raw.startsWith('@')) {
      const match = raw.match(HEADER);
      if (!match) { diagnostic(diagnostics, index + 1, 1, 'invalid @section header'); continue; }
      if (current) current.end = index;
      const global = match[2] === ':' || GLOBALS.has(match[1]);
      if (sections.some(section => section.name === match[1])) {
        diagnostic(diagnostics, index + 1, 1, `duplicate @${match[1]} section`);
      }
      current = { name: match[1], global, line: index + 1, end: lines.length };
      sections.push(current);
    } else if (!current && raw.trim()) {
      diagnostic(diagnostics, index + 1, 1, 'expected @circuit or @UnitName before content');
    }
  }
  const views: GraphView[] = [];
  const globals: ParameterBlock[] = [];
  const settings = new Map<string, ParsedMapping>();
  for (const section of sections) {
    if (!section.global) {
      views.push(parseView(section, lines, diagnostics));
    } else {
      const chunk = lines.slice(section.line, section.end);
      const source = lines.slice(section.line - 1, section.end).join('\n');
      globals.push({ id: '@' + section.name, name: section.name, type: 'global', source,
        range: { start: loc(section.line), end: loc(section.end, (lines[section.end - 1] || '').length + 1) },
        children: [], });
      settings.set(section.name, parseYamlMapping(chunk.map((value, index) => ({
        text: value, line: section.line + 1 + index,
      })), diagnostics));
    }
  }
  // A unit definition inside @UnitName describes the unit in the parent circuit.
  const circuit = views.find(view => view.name === 'circuit');
  if (circuit) for (const childView of views) {
    if (childView.name === 'circuit') continue;
    const unit = circuit.nodes.find(node => node.name === childView.name);
    const definition = childView.parameters.find(block => block.name === childView.name);
    if (unit && definition) {
      const [type, controllerType] = definition.type.split(',').map(value => value.trim());
      unit.type = type;
      unit.controllerType = controllerType;
      unit.parameterBlock = definition.id;
    }
  }
  const simulation = settings.get('simulation')?.value || {};
  const duration = typeof simulation.t_end === 'number' ? simulation.t_end : NaN;
  const initialState = simulation.initial && typeof simulation.initial === 'object' &&
    !Array.isArray(simulation.initial) ? simulation.initial as Mapping : {};
  const initial = typeof initialState.t === 'number' ? initialState.t : 0;
  const eventsSection = sections.find(section => section.name === 'events');
  const events = readEvents(settings.get('events'), diagnostics, eventsSection?.line || 1);
  const defaultView = circuit ? 'circuit' : views[0]?.name || '';
  if (!views.length) diagnostic(diagnostics, 1, 1, 'no schematic view found (expected @circuit)');
  return { views, globals, events, timeline: buildTimeline(events, duration, initial),
    diagnostics, defaultView };
}
