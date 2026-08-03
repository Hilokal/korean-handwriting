import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { nodePolyfills } from "vite-plugin-node-polyfills";

// web_pen_sdk requires the Node `zlib` module and the `Buffer` global in the
// browser; the extra includes cover browserify-zlib's own transitive needs.
export default defineConfig({
  root: "client",
  // web_pen_sdk imports .nproj XML files expecting URL strings (CRA file-loader
  // behavior); treat them as assets so Rollup doesn't parse them as JS.
  assetsInclude: ["**/*.nproj"],
  plugins: [
    react(),
    nodePolyfills({
      include: ["zlib", "buffer", "stream", "util", "assert", "process"],
      globals: { Buffer: true, process: true },
    }),
  ],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    port: 3000,
    proxy: {
      "/api": "http://localhost:8080",
    },
  },
});
