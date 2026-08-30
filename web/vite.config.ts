import { createReadStream, existsSync } from "node:fs";
import { basename } from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig, type Plugin } from "vite";
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

/**
 * Serve the ONNX runtime's own files byte for byte in development.
 *
 * The runtime loads its worker module with a dynamic `import()`. Vite's dev
 * server rewrites dynamic imports and appends `?import`, then tries to run the
 * result through its transform pipeline -- but these are prebuilt files in
 * `public/`, not source, and the transform fails:
 *
 *     Failed to fetch dynamically imported module:
 *     /ort/ort-wasm-simd-threaded.mjs?import
 *
 * On the page that surfaces as "no available backend found". Inside the search
 * worker it does not surface at all: the page sits at "Loading the agent…"
 * forever with nothing in the console.
 *
 * None of this happens in a production build, because a static file server has
 * no transform pipeline -- which is exactly why it went unnoticed. `preview`
 * and the Playwright suite were fine the whole time.
 */
function serveRuntimeVerbatim(): Plugin {
  return {
    name: "serve-ort-verbatim",
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const url = request.url ?? "";
        if (!url.startsWith("/ort/")) return next();

        // basename only: this reads from a directory decided here, never from
        // a path a request can steer.
        const name = basename(url.split("?")[0] ?? "");
        const file = fileURLToPath(new URL(`./public/ort/${name}`, import.meta.url));
        if (!existsSync(file)) return next();

        response.setHeader(
          "Content-Type",
          name.endsWith(".wasm") ? "application/wasm" : "text/javascript",
        );
        createReadStream(file).pipe(response);
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), serveRuntimeVerbatim(), dropUnusedRuntimeBinary()],

  // The runtime is excluded from dependency pre-bundling: its .wasm and worker
  // module are fetched by URL from /ort/ (see serveRuntimeVerbatim above), not
  // imported as source, and pre-bundling them only gets in the way.
  optimizeDeps: { exclude: ["onnxruntime-web"] },

  worker: { format: "es" },

  build: {
    // Vite would normally clear dist/ before each build. That fights the
    // development loop, which runs `vite build --watch` and `vite preview`
    // together: preview serves files out of dist/ while the watch build wants to
    // delete them, and on Windows an open handle makes the delete fail outright
    // rather than wait. The result was a build that died with EPERM on every
    // save.
    //
    // Clearing is an explicit step in the `build` script instead, so the
    // production build stays deterministic and the watch loop stops fighting the
    // server sitting on its output.
    emptyOutDir: false,

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
