import { useEffect, useRef } from "react";
import { Dot } from "../lib/generator";

// Live pen view. Points stream into `store.dots` (a mutable ref shared with
// App) while generation runs; this component reveals them at pen speed with
// its own rAF loop, drawing directly into an <svg> via the DOM to avoid a
// React render per frame. Bump `playNonce` to restart playback from point 0.

export interface DotStore {
  dots: Dot[];
}

const POINTS_PER_SEC = 120; // reveal rate; pen recordings are ~50 pts/s
const PAD = 0.6;
const MIN_SPAN = 4; // don't zoom in absurdly on the first few points

export default function HandwritingCanvas({
  store,
  playNonce,
  onFullyDrawn,
}: {
  store: DotStore;
  playNonce: number;
  onFullyDrawn?: () => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const doneRef = useRef(false);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    doneRef.current = false;

    let raf = 0;
    const start = performance.now();

    const render = (now: number) => {
      raf = requestAnimationFrame(render);
      const dots = store.dots;
      const reveal = Math.min(
        dots.length,
        Math.floor(((now - start) / 1000) * POINTS_PER_SEC),
      );

      // Absolute points + bounds over EVERYTHING generated so far, so the
      // viewBox settles as soon as generation (which outruns the pen) does.
      let x = 0,
        y = 0,
        minX = 0,
        minY = 0,
        maxX = 0,
        maxY = 0;
      const pts: { x: number; y: number; pen: number }[] = [];
      for (let i = 0; i < dots.length; i++) {
        x += dots[i].x;
        y += dots[i].y;
        if (i < reveal) pts.push({ x, y, pen: dots[i].penState });
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }

      const spanX = Math.max(maxX - minX, MIN_SPAN);
      const spanY = Math.max(maxY - minY, MIN_SPAN);
      svg.setAttribute(
        "viewBox",
        `${(minX - PAD).toFixed(2)} ${(minY - PAD).toFixed(2)} ` +
          `${(spanX + 2 * PAD).toFixed(2)} ${(spanY + 2 * PAD).toFixed(2)}`,
      );

      // Pen tip ~7% of line height (see animatedSvg.ts); width-relative
      // sizing over-inks long lines.
      const strokeWidth = Math.max((maxY - minY) * 0.07, spanX * 0.008);
      const parts: string[] = [];
      let d = "";
      let newStroke = true;
      for (const p of pts) {
        d += `${newStroke ? "M" : "L"} ${p.x.toFixed(2)} ${p.y.toFixed(2)} `;
        newStroke = p.pen >= 0.5;
      }
      if (d) {
        parts.push(
          `<path d="${d}" fill="none" stroke="currentColor" ` +
            `stroke-width="${strokeWidth.toFixed(3)}" ` +
            `stroke-linecap="round" stroke-linejoin="round"/>`,
        );
      }
      svg.innerHTML = parts.join("");

      if (!doneRef.current && dots.length > 0 && reveal >= dots.length) {
        // May still be mid-generation; App only treats this as final when
        // generation has also finished.
        doneRef.current = true;
        onFullyDrawn?.();
      } else if (reveal < dots.length) {
        doneRef.current = false;
      }
    };

    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, [store, playNonce, onFullyDrawn]);

  return <svg ref={svgRef} className="pen-canvas" viewBox="0 0 20 5" />;
}
