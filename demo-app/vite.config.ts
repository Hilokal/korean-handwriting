import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, Plugin } from "vite";
import { HOME, ROUTES, SITE_URL } from "./src/routes";

const here = path.dirname(fileURLToPath(import.meta.url));

const escapeHtml = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");

// Link-preview crawlers don't run JS and Cloudflare serves one index.html for
// every path, so emit a copy per route with that route's title/description/
// og:* swapped in. Written as dist/blog/hangul-day.html (not .../index.html)
// so Cloudflare's auto-trailing-slash serves /blog/hangul-day without a
// redirect. The app itself still routes on location.pathname.
function prerenderRoutes(): Plugin {
  return {
    name: "prerender-routes",
    apply: "build",
    closeBundle() {
      const dist = path.join(here, "dist");
      const base = fs.readFileSync(path.join(dist, "index.html"), "utf8");
      for (const r of ROUTES) {
        if (r.path === HOME.path) continue;
        const url = `${SITE_URL}${r.path}`;
        const html = base
          .replace(/<title>[^<]*<\/title>/, `<title>${escapeHtml(r.title)}</title>`)
          .replace(/(<meta name="description" content=")[^"]*/, `$1${escapeHtml(r.description)}`)
          .replace(/(<meta property="og:title" content=")[^"]*/, `$1${escapeHtml(r.title)}`)
          .replace(/(<meta property="og:description" content=")[^"]*/, `$1${escapeHtml(r.description)}`)
          .replace(/(<meta property="og:type" content=")[^"]*/, `$1${r.ogType}`)
          .replace(/(<meta property="og:url" content=")[^"]*/, `$1${url}`)
          .replace(/(<link rel="canonical" href=")[^"]*/, `$1${url}`);
        if (html === base) throw new Error(`prerender: no meta replaced for ${r.path}`);
        const out = path.join(dist, `${r.path.replace(/^\//, "")}.html`);
        fs.mkdirSync(path.dirname(out), { recursive: true });
        fs.writeFileSync(out, html);
      }
    },
  };
}

// onnxruntime-web loads its wasm backend at runtime from a URL, not via the
// bundler. Copy the wasm (+ loader .mjs) files into public/ort/ so they are
// served in dev and shipped in dist/, matching ort.env.wasm.wasmPaths = "/ort/".
function copyOrtWasm(): Plugin {
  return {
    name: "copy-ort-wasm",
    buildStart() {
      const dist = path.join(here, "node_modules", "onnxruntime-web", "dist");
      const out = path.join(here, "public", "ort");
      fs.mkdirSync(out, { recursive: true });
      for (const f of fs.readdirSync(dist)) {
        // Only the plain simd-threaded backend — not jsep (WebGPU, 27 MB),
        // asyncify, or jspi, which we never request.
        if (/^ort-wasm-simd-threaded\.(wasm|mjs)$/.test(f)) {
          fs.copyFileSync(path.join(dist, f), path.join(out, f));
        }
      }
    },
    // ort's ESM references its wasm via new URL(import.meta.url), so Rollup
    // emits a second 13 MB copy into assets/. The runtime only ever fetches
    // from wasmPaths (/ort/), so drop the duplicate from the bundle.
    generateBundle(_opts, bundle) {
      for (const name of Object.keys(bundle)) {
        if (/ort-wasm.*\.wasm$/.test(name)) delete bundle[name];
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), copyOrtWasm(), prerenderRoutes()],
  server: { port: 5174 },
});
