/** Orthogonal edge routing between rectangular blocks. */
export type Side = 'left' | 'right' | 'top' | 'bottom';
export interface Point { x: number; y: number }
export interface Box { x: number; y: number; width: number; height: number }
export interface Anchor extends Point { side: Side; box: Box }

const GAP = 28;

export function chooseSides(from: Box, to: Box, directed: boolean): [Side, Side] {
  const dx = to.x - from.x, dy = to.y - from.y;
  if (!directed && Math.abs(dy) > Math.abs(dx) * 1.35) {
    return dy >= 0 ? ['bottom', 'top'] : ['top', 'bottom'];
  }
  return dx >= 0 ? ['right', 'left'] : ['left', 'right'];
}

function outward(point: Anchor, distance = GAP): Point {
  if (point.side === 'left') return { x: point.x - distance, y: point.y };
  if (point.side === 'right') return { x: point.x + distance, y: point.y };
  if (point.side === 'top') return { x: point.x, y: point.y - distance };
  return { x: point.x, y: point.y + distance };
}

/** Return a Manhattan polyline; every segment is horizontal or vertical. */
export function routeEdge(from: Anchor, to: Anchor): Point[] {
  if (from.x === to.x || from.y === to.y) return [from, to];
  const start = outward(from), end = outward(to);
  const horizontal = from.side === 'left' || from.side === 'right';
  const targetHorizontal = to.side === 'left' || to.side === 'right';
  if (horizontal && targetHorizontal) {
    const facing = (from.side === 'right' && to.side === 'left' && start.x <= end.x) ||
      (from.side === 'left' && to.side === 'right' && start.x >= end.x);
    if (facing) {
      const middle = (start.x + end.x) / 2;
      return [from, start, { x: middle, y: start.y }, { x: middle, y: end.y }, end, to];
    }
    const outside = from.side === 'right' ?
      Math.max(from.box.x + from.box.width / 2, to.box.x + to.box.width / 2) + GAP :
      Math.min(from.box.x - from.box.width / 2, to.box.x - to.box.width / 2) - GAP;
    return [from, { x: outside, y: from.y }, { x: outside, y: to.y }, to];
  }
  if (!horizontal && !targetHorizontal) {
    const middle = (start.y + end.y) / 2;
    return [from, start, { x: start.x, y: middle }, { x: end.x, y: middle }, end, to];
  }
  return horizontal ? [from, start, { x: end.x, y: start.y }, end, to] :
    [from, start, { x: start.x, y: end.y }, end, to];
}

const rounded = (value: number) => Number(value.toFixed(2));

export function pathData(points: readonly Point[]): string {
  return points.map((point, index) =>
    `${index ? 'L' : 'M'} ${rounded(point.x)} ${rounded(point.y)}`).join(' ');
}
