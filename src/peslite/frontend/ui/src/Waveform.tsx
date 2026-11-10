import { useEffect, useMemo, useRef, useState, type PointerEvent, type WheelEvent } from 'react';
import type { CsvResponse } from './transport';

const COLORS = ['#125e94', '#b15c29', '#1b8272', '#8b57ad', '#a65b73', '#687b27'];
const numeric = (value: number | string | null): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

type Trace = { id: string; label: string; times: (number | null)[]; values: (number | null)[] };

/** Traces from different CSV files retain their own time vectors and sample rates. */
function traces(data: CsvResponse, prefix: string): Trace[] {
  const timeIndex = data.columns.findIndex(name => /^(t|time|time_s)$/i.test(name));
  const t = timeIndex < 0 ? 0 : timeIndex;
  const times = data.rows.map(row => numeric(row[t]));
  return data.columns.flatMap((name, index) => index === t ? [] : [{
    id: `${prefix}:${name}`, label: prefix ? `${prefix} · ${name}` : name,
    times, values: data.rows.map(row => numeric(row[index])),
  }]);
}

/** Canvas waveform plot with bounded data, independent time axes and point inspection. */
export function Waveform({ data, comparison }: {
  data: CsvResponse;
  comparison?: { path: string; data: CsvResponse } | null;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [domain, setDomain] = useState<[number, number] | null>(null);
  const [cursor, setCursor] = useState<{ x: number; values: string[] } | null>(null);
  const drag = useRef<{ x: number; domain: [number, number] } | null>(null);
  const all = useMemo(() => [
    ...traces(data, ''),
    ...(comparison ? traces(comparison.data, comparison.path) : []),
  ], [data, comparison]);
  const active = selected.length ? selected.filter(id => all.some(trace => trace.id === id)) : [];
  const visible = active.length ? all.filter(trace => active.includes(trace.id)) : all.slice(0, 3);
  let start = Infinity;
  let end = -Infinity;
  for (const trace of all) for (const time of trace.times) {
    if (time !== null) { start = Math.min(start, time); end = Math.max(end, time); }
  }
  const extent: [number, number] = start === Infinity ? [0, 1] : [start, end];
  const xRange: [number, number] = domain ?? extent;

  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    function paint() {
      if (!element) return;
      const width = Math.max(320, element.clientWidth || 800);
      const height = 330;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      element.width = Math.floor(width * ratio);
      element.height = Math.floor(height * ratio);
      const ctx = element.getContext('2d');
      if (!ctx) return;
      ctx.scale(ratio, ratio);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      const box = { left: 62, top: 14, right: width - 16, bottom: height - 34 };
      let lower = Infinity;
      let upper = -Infinity;
      for (const trace of visible) for (let index = 0; index < trace.times.length; index++) {
        const t = trace.times[index], y = trace.values[index];
        if (t !== null && y !== null && t >= xRange[0] && t <= xRange[1]) {
          lower = Math.min(lower, y); upper = Math.max(upper, y);
        }
      }
      let minY = lower === Infinity ? -1 : lower;
      let maxY = upper === -Infinity ? 1 : upper;
      if (minY === maxY) { minY -= 1; maxY += 1; }
      const tx = (t: number) => box.left + (t - xRange[0]) /
        (xRange[1] - xRange[0] || 1) * (box.right - box.left);
      const ty = (y: number) => box.bottom - (y - minY) /
        (maxY - minY) * (box.bottom - box.top);
      ctx.fillStyle = '#698093';
      ctx.strokeStyle = '#e3ebf2';
      ctx.font = '11px system-ui';
      ctx.textAlign = 'right';
      for (let i = 0; i <= 4; i++) {
        const y = box.top + i * (box.bottom - box.top) / 4;
        ctx.beginPath(); ctx.moveTo(box.left, y); ctx.lineTo(box.right, y); ctx.stroke();
        ctx.fillText((maxY - i * (maxY - minY) / 4).toPrecision(4), box.left - 9, y + 4);
      }
      ctx.textAlign = 'center';
      for (let i = 0; i <= 5; i++) {
        const x = box.left + i * (box.right - box.left) / 5;
        ctx.fillText((xRange[0] + i * (xRange[1] - xRange[0]) / 5).toPrecision(4), x, height - 13);
      }
      visible.forEach((trace, color) => {
        ctx.strokeStyle = COLORS[color % COLORS.length];
        ctx.lineWidth = 1.65;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < trace.times.length; i++) {
          const t = trace.times[i], y = trace.values[i];
          if (t === null || y === null || t < xRange[0] || t > xRange[1]) {
            started = false;
          } else if (started) {
            ctx.lineTo(tx(t), ty(y));
          } else {
            ctx.moveTo(tx(t), ty(y));
            started = true;
          }
        }
        ctx.stroke();
      });
      if (cursor) {
        ctx.strokeStyle = '#a3b7c8';
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(cursor.x, box.top);
        ctx.lineTo(cursor.x, box.bottom); ctx.stroke();
      }
    }
    paint();
    const observer = new ResizeObserver(paint);
    observer.observe(element);
    return () => observer.disconnect();
  }, [data, comparison, visible.map(trace => trace.id).join('|'), xRange[0], xRange[1], cursor?.x]);

  function onMove(event: PointerEvent<HTMLCanvasElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const scale = Math.max(bounds.width, 320) - 78;
    const time = xRange[0] + (x - 62) / scale * (xRange[1] - xRange[0]);
    if (drag.current) {
      const move = (event.clientX - drag.current.x) / scale *
        (drag.current.domain[1] - drag.current.domain[0]);
      setDomain([drag.current.domain[0] - move, drag.current.domain[1] - move]);
    }
    const values: string[] = [`t ≈ ${time.toPrecision(6)}`];
    for (const trace of visible) {
      let nearest = -1;
      let error = Infinity;
      for (let index = 0; index < trace.times.length; index++) {
        const current = trace.times[index];
        if (current === null) continue;
        const distance = Math.abs(current - time);
        if (distance < error) { nearest = index; error = distance; }
      }
      if (nearest !== -1) {
        const value = trace.values[nearest];
        values.push(`${trace.label}: ${value === null ? '—' : value.toPrecision(6)}`);
      }
    }
    setCursor({ x, values });
  }

  function onWheel(event: WheelEvent<HTMLCanvasElement>) {
    event.preventDefault();
    const span = xRange[1] - xRange[0] || 1;
    const scale = event.deltaY < 0 ? 0.82 : 1.22;
    const center = (xRange[0] + xRange[1]) / 2;
    setDomain([center - span * scale / 2, center + span * scale / 2]);
  }
  return <div className="waveform">
    <div className="plot-toolbar">
      <strong>Waveforms</strong>
      <button onClick={() => setDomain(null)}>Fit</button>
      <span>Wheel to zoom · drag to pan · hover to inspect</span>
    </div>
    <div className="signal-options">{all.map(trace =>
      <label key={trace.id}><input type="checkbox" checked={visible.includes(trace)}
        onChange={() => setSelected(old => {
          const current = old.length ? old : all.slice(0, 3).map(item => item.id);
          return current.includes(trace.id) ? current.filter(item => item !== trace.id)
            : [...current, trace.id];
        })} />{trace.label}</label>)}</div>
    <canvas ref={canvas} className="wave-canvas" onPointerMove={onMove}
      onPointerDown={(event: PointerEvent<HTMLCanvasElement>) => {
        drag.current = { x: event.clientX, domain: xRange };
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerUp={() => { drag.current = null; }}
      onPointerCancel={() => { drag.current = null; }}
      onWheel={onWheel} />
    <div className="plot-values">{cursor?.values.join('  ·  ') || 'Move over a trace to inspect values'}</div>
    <div className="caption">{data.rows.length} displayed samples · missing/NaN = gaps · each CSV has its own time axis.</div>
  </div>;
}
