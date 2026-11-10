import type {
  GraphGroup, GraphLayout, GraphView, LayoutGroup, LayoutNode, NodeOffset, PreviewLayout,
} from './types.js';

const NODE_WIDTH = 142;
const NODE_HEIGHT = 64;
const COLUMN_STEP = 222;
const ROW_STEP = 122;
const MARGIN_X = 130;
const MARGIN_Y = 114;
const MAX_OFFSET = 1e6;

export const LAYOUT_FORMAT = 'peslite.preview-layout';

export function emptyLayout(): PreviewLayout {
  return { format: LAYOUT_FORMAT, version: 1, views: {} };
}

/** Drop malformed or stale entries before a host-provided layout reaches the renderer. */
export function normalizeLayout(value: unknown): PreviewLayout {
  const result = emptyLayout();
  if (!value || typeof value !== 'object') return result;
  const record = value as { format?: unknown; version?: unknown; views?: unknown };
  if (record.format !== undefined && record.format !== LAYOUT_FORMAT) return result;
  if (record.version !== undefined && record.version !== 1) return result;
  if (!record.views || typeof record.views !== 'object') return result;
  for (const [view, nodes] of Object.entries(record.views as Record<string, unknown>)) {
    if (!view || !nodes || typeof nodes !== 'object') continue;
    const offsets: Record<string, NodeOffset> = {};
    for (const [node, raw] of Object.entries(nodes as Record<string, unknown>)) {
      const offset = raw as { dx?: unknown; dy?: unknown } | null;
      const dx = offset?.dx, dy = offset?.dy;
      if (!node || typeof dx !== 'number' || typeof dy !== 'number' ||
          !Number.isFinite(dx) || !Number.isFinite(dy) ||
          Math.abs(dx) > MAX_OFFSET || Math.abs(dy) > MAX_OFFSET) continue;
      if (dx || dy) offsets[node] = { dx, dy };
    }
    if (Object.keys(offsets).length) result.views[view] = offsets;
  }
  return result;
}

/** Text-order graph layout, calculated only from the view, never from an event interval. */
export function layoutView(view: GraphView,
                           offsets: Readonly<Record<string, NodeOffset>> = {}): GraphLayout {
  const slots = new Map<string, { col: number; row: number }>();
  const occupied = new Set<string>();
  let lastRow = -1;
  function assign(id: string, col: number, row: number): void {
    col = Math.max(0, col);
    row = Math.max(0, row);
    while (occupied.has(`${col}:${row}`)) row++;
    occupied.add(`${col}:${row}`);
    slots.set(id, { col, row });
    lastRow = Math.max(lastRow, row);
  }
  // A first path establishes the horizontal line, and later branches are
  // placed beside known nodes without shifting any previously assigned node.
  for (const edge of view.edges) {
    let from = slots.get(edge.from);
    let to = slots.get(edge.to);
    if (!from && !to) {
      assign(edge.from, 0, lastRow < 0 ? 0 : lastRow + 1);
      from = slots.get(edge.from);
    }
    if (from && !to) assign(edge.to, from.col + 1, from.row);
    if (!from && to) assign(edge.from, Math.max(0, to.col - 1), to.row);
  }
  for (const node of view.nodes) if (!slots.has(node.id)) assign(node.id, 0, lastRow + 1);
  const nodes = new Map<string, LayoutNode>();
  for (const [id, slot] of slots) {
    const offset = offsets[id];
    nodes.set(id, {
      id, x: MARGIN_X + slot.col * COLUMN_STEP + (offset?.dx || 0),
      y: MARGIN_Y + slot.row * ROW_STEP + (offset?.dy || 0),
      width: NODE_WIDTH, height: NODE_HEIGHT,
    });
  }
  const groups = new Map<string, LayoutGroup>();
  const groupMap = new Map(view.groups.map(group => [group.id, group]));
  function members(group: GraphGroup, seen: Set<string>): LayoutNode[] {
    if (seen.has(group.id)) return [];
    const next = new Set(seen).add(group.id);
    return group.members.flatMap(member => {
      if (member.startsWith('&')) {
        const child = groupMap.get(member);
        return child ? members(child, next) : [];
      }
      const point = nodes.get(member);
      return point ? [point] : [];
    });
  }
  function depth(group: GraphGroup, seen: Set<string>): number {
    if (seen.has(group.id)) return 0;
    const next = new Set(seen).add(group.id);
    return 1 + Math.max(0, ...group.members.map(member => {
      const child = groupMap.get(member);
      return child ? depth(child, next) : 0;
    }));
  }
  // Larger groups are painted first so nested groups are visually enclosed.
  for (const group of view.groups) {
    const positions = members(group, new Set());
    if (!positions.length) continue;
    const deep = depth(group, new Set());
    const padding = 24 + deep * 13;
    const left = Math.min(...positions.map(item => item.x - item.width / 2)) - padding;
    const right = Math.max(...positions.map(item => item.x + item.width / 2)) + padding;
    const top = Math.min(...positions.map(item => item.y - item.height / 2)) - padding - 10;
    const bottom = Math.max(...positions.map(item => item.y + item.height / 2)) + padding;
    groups.set(group.id, { id: group.id, x: left, y: top,
      width: right - left, height: bottom - top, depth: deep });
  }
  const boxes = [
    ...[...nodes.values()].map(item => ({ x: item.x - item.width / 2,
      y: item.y - item.height / 2, width: item.width, height: item.height })),
    ...groups.values(),
  ];
  const minX = Math.min(0, ...boxes.map(item => item.x));
  const minY = Math.min(0, ...boxes.map(item => item.y));
  const left = minX < 0 ? minX - 40 : 0;
  const top = minY < 0 ? minY - 40 : 0;
  return { nodes, groups, left, top,
    width: Math.max(600, ...boxes.map(item => item.x + item.width)) + 90 - left,
    height: Math.max(340, ...boxes.map(item => item.y + item.height)) + 90 - top,
  };
}
