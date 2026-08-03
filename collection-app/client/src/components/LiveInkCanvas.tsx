import { useEffect, useRef } from "react";
import { Point } from "./strokeSvg";

// Auto-fit: Ncode coordinates are absolute paper positions, so we don't know
// where on the page the worker writes. Fit the cleaned points' bounding box
// into the canvas on every update. A full clear+redraw per dot is cheap at
// chunk scale (a few thousand points) and stays correct when cleaning
// retroactively changes earlier points (e.g. the majority page flips early
// in a session and off-page dots disappear).

const PAD = 16;
const MAX_SCALE = 80; // px per Ncode unit; prevents absurd zoom on the first dot
const LINE_WIDTH = 2;

export function LiveInkCanvas({ points }: { points: Point[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth;
    const cssHeight = canvas.clientHeight;
    if (canvas.width !== cssWidth * dpr || canvas.height !== cssHeight * dpr) {
      canvas.width = cssWidth * dpr;
      canvas.height = cssHeight * dpr;
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);
    if (points.length === 0) return;

    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const p of points) {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    }
    const width = maxX - minX || 0.5;
    const height = maxY - minY || 0.5;
    const scale = Math.min(
      (cssWidth - 2 * PAD) / width,
      (cssHeight - 2 * PAD) / height,
      MAX_SCALE,
    );
    const ox = PAD + (cssWidth - 2 * PAD - width * scale) / 2 - minX * scale;
    const oy = PAD + (cssHeight - 2 * PAD - height * scale) / 2 - minY * scale;

    ctx.lineWidth = LINE_WIDTH;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#222";
    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const x = p.x * scale + ox;
      const y = p.y * scale + oy;
      if (i === 0 || p.startNew) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, [points]);

  return <canvas ref={canvasRef} className="live-canvas" />;
}
