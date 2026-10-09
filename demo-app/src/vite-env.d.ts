/// <reference types="vite/client" />

// Injected by vite.config.ts `define`: the onnxruntime-web package version,
// which is also the directory the wasm runtime is served from (/ort/<ver>/).
declare const __ORT_VERSION__: string;
