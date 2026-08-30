import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

/**
 * Cross-origin isolation, which is what multi-threaded WebAssembly needs.
 *
 * This is not optional here. Measured on the trained network, a single thread
 * does about 160 positions per second in the browser, so the top difficulty --
 * 800 simulations -- takes five and a half seconds. Throughput is flat across
 * batch sizes, so the cost is arithmetic rather than call overhead and batching
 * search leaves would buy nothing. More cores is the only lever available, and
 * these two headers are what unlock it.
 *
 * `public/_headers` sets the same pair on the deployed site. Cloudflare Pages
 * reads that file; GitHub Pages cannot set headers at all, which is the concrete
 * reason it was not chosen.
 */
const CROSS_ORIGIN_ISOLATION = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

/**
 * Drop the bundler's own copy of the runtime binary.
 *
 * `onnxruntime-web` imports its `.wasm` as a URL, so the bundler dutifully
 * emits a content-hashed copy of it. Nothing fetches that copy: the runtime is
 * pointed at `/ort/` instead, because its worker threads need a predictable
 * path (see scripts/stage-runtime.mjs). Left in place it is fourteen megabytes
 * of build output that is never served, and a second answer to "which binary is
 * this site actually running".
 */
function dropUnusedRuntimeBinary() {
  return {
    name: "drop-unused-ort-wasm",
    generateBundle(_options: unknown, bundle: Record<string, unknown>) {
      for (const name of Object.keys(bundle)) {
        if (name.includes("ort-wasm") && name.endsWith(".wasm")) delete bundle[name];
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), dropUnusedRuntimeBinary()],

  // onnxruntime-web ships .wasm files that must be served, not bundled into a
  // JavaScript chunk. Excluding it from dependency pre-bundling is what keeps
  // the bundler treating them as assets.
  optimizeDeps: { exclude: ["onnxruntime-web"] },

  worker: { format: "es" },

  build: {
    rollupOptions: {
      input: {
        // The app, and a benchmark page. The benchmark is a real part of the
        // project rather than a scratch file: the decision to run the agent in
        // the browser rests on how fast it actually is on a given device, and
        // that is not something to take on trust from one laptop.
        main: fileURLToPath(new URL("./index.html", import.meta.url)),
        bench: fileURLToPath(new URL("./bench/index.html", import.meta.url)),
      },
    },
  },

  server: { headers: CROSS_ORIGIN_ISOLATION },
  preview: { headers: CROSS_ORIGIN_ISOLATION },
});
