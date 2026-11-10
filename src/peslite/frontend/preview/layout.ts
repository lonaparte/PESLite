import type { GraphGroup, GraphLayout, GraphView, LayoutGroup, LayoutNode } from './types.js';

const NODE_WIDTH = 142;
const NODE_HEIGHT = 64;
const COLUMN_STEP = 222;
const ROW_STEP = 122;
const MARGIN_X = 130;
const MARGIN_Y = 114;

/** Text-order graph layout, calculated only from the view, never from an event interval. */
export function layoutView(view: GraphView): GraphLayout {
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
  for (const [id, slot] of slots) nodes.set(id, {
    id, x: MARGIN_X + slot.col * COLUMN_STEP, y: MARGIN_Y + slot.row * ROW_STEP,
    width: NODE_WIDTH, height: NODE_HEIGHT,
  });
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
  const allRights = [...nodes.values()].map(item => item.x + item.width / 2);
  const allBottoms = [...nodes.values()].map(item => item.y + item.height / 2);
  const groupRights = [...groups.values()].map(item => item.x + item.width);
  const groupBottoms = [...groups.values()].map(item => item.y + item.height);
  return { nodes, groups,
    width: Math.max(600, ...allRights, ...groupRights) + 90,
    height: Math.max(340, ...allBottoms, ...groupBottoms) + 90,
  };
}
