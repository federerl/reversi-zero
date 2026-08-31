/**
 * Copy the ONNX runtime's own files into the site before building.
 *
 * The runtime needs two files at a path it can predict: the WebAssembly binary,
 * and the small module it starts its worker threads from. Letting the bundler
 * place them instead looks like it works -- until threads are switched on, at
 * which point the worker cannot find the module the bundler renamed, and the
 * page hangs at load with no error in the console at all. That is why
 * `src/engine/onnx.ts` sets `wasmPaths` to a fixed directory and this script
 * fills it.
 *
 * Only the SIMD build is copied. `onnxruntime-web` also ships a WebGPU-capable
 * one at 27.8 MB against 14 MB, and the site does not use WebGPU: for a network
 * this small, dispatching each operation to a GPU generally costs more than the
 * arithmetic it saves.
 */

import { copyFileSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const here = dirname(fileURLToPath(import.meta.url));
const from = join(here, "..", "node_modules", "onnxruntime-web", "dist");
const to = join(here, "..", "public", "ort");

const NEEDED = [
  "ort-wasm-simd-threaded.wasm", // the runtime itself
  "ort-wasm-simd-threaded.mjs", // what it starts its threads from
];

mkdirSync(to, { recursive: true });

let total = 0;
for (const name of NEEDED) {
  const source = join(from, name);
  copyFileSync(source, join(to, name));
  const compressed = gzipSync(readFileSync(source)).length;
  total += compressed;
  console.log(
    `  ${name.padEnd(32)} ${(statSync(source).size / 1e6).toFixed(1).padStart(5)} MB raw` +
      `  ${(compressed / 1e6).toFixed(2).padStart(5)} MB gzipped`,
  );
}
console.log(`  ${"".padEnd(32)} ${"".padStart(5)}        ${(total / 1e6).toFixed(2).padStart(5)} MB downloaded once, then cached`);
