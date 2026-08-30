import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],

  // onnxruntime-web ships .wasm files that must be served, not bundled. Vite
  // treats them as assets when they are excluded from dependency pre-bundling.
  optimizeDeps: { exclude: ["onnxruntime-web"] },

  worker: { format: "es" },

  server: {
    headers: {
      // Cross-origin isolation, which is what multi-threaded WebAssembly needs.
      // We do not need threads today -- a search is comfortable single-threaded
      // at this network size -- but developing without these headers and adding
      // them later means discovering in production that something else on the
      // page stopped loading under them.
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
