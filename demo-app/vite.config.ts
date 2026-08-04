import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, Plugin } from "vite";

const here = path.dirname(fileURLToPath(import.meta.url));

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
  plugins: [react(), copyOrtWasm()],
  server: { port: 5174 },
});
