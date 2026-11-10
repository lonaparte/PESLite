import type { EventSegment } from './types.js';

/** The interval at `time`; events at a boundary already apply at that instant. */
export function segmentAt(timeline: readonly EventSegment[], time: number): number {
  let low = 0, high = timeline.length - 1, found = 0;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (timeline[middle].start <= time + 1e-12) {
      found = middle;
      low = middle + 1;
    } else high = middle - 1;
  }
  return found;
}

/** Round tick values covering the requested range. */
export function niceTicks(start: number, end: number, target = 8): number[] {
  const span = end - start;
  if (!Number.isFinite(span) || span <= 0) return [start];
  const raw = span / Math.max(1, target);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(value => value * magnitude)
    .find(value => value >= raw * (1 - 1e-9))!;
  const ticks: number[] = [];
  for (let index = Math.ceil(start / step - 1e-9);
       index * step <= end + step * 1e-9; index++) {
    ticks.push(Number((index * step).toPrecision(12)));
  }
  return ticks;
}
