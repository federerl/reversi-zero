/**
 * How fast is the agent on *this* device?
 *
 * The design for the browser engine rested on a projection: ONNX Runtime does a
 * forward pass in 0.30 ms natively, WebAssembly is typically 2-4x slower than
 * native, so a search should cost about a millisecond per simulation. The first
 * real measurement came out at nearly seven, which is the sort of gap that
 * decides whether the top difficulty is usable.
 *
 * So this measures rather than argues, and separates the two things that could
 * be responsible:
 *
 *   **network**  one forward pass, nothing else. If this is slow, the runtime is
 *                the cost and batching or threads are the answers.
 *   **search**   a whole move. The difference between this and `network` times
 *                the simulation count is what the tree and the marshalling
 *                between JavaScript and WebAssembly cost.
 *
 * Open `/bench/` on the device in question. It prints a table.
 */

import modelsManifest from "../src/engine/models.json";

import { encodeBatch } from "../src/engine/features";
import { MCTS } from "../src/engine/mcts";
import { loadModel, type ModelDescriptor } from "../src/engine/onnx";
import { apply, initialState, legalActions, type State } from "../src/engine/rules";

const MODELS = modelsManifest.models as unknown as ModelDescriptor[];
const out = document.getElementById("out")!;

function log(line: string): void {
  out.textContent += line + "\n";
}

/** A handful of real mid-game positions, so the numbers reflect real branching. */
function positions(count: number): State[] {
  const found: State[] = [];
  let state = initialState(8);
  let seed = 12345;
  const next = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff);

  while (found.length < count) {
    const legal = legalActions(state);
    if (legal.length === 0) {
      state = initialState(8);
      continue;
    }
    state = apply(state, legal[next() % legal.length]!);
    if (legalActions(state).length >= 2) found.push(state);
  }
  return found;
}

async function main(): Promise<void> {
  const descriptor = MODELS[0]!;
  log(`device:  ${navigator.hardwareConcurrency ?? "?"} logical cores`);
  log(`agent:   ${descriptor.label}`);
  log(`loading ${descriptor.url} …\n`);

  const started = performance.now();
  const evaluator = await loadModel(descriptor);
  log(`model ready in ${Math.round(performance.now() - started)} ms\n`);

  const boards = positions(64);

  // ---- the network on its own ----
  log("one forward pass");
  log("  batch   ms/call   positions/s");
  for (const batch of [1, 4, 8, 16, 32]) {
    const slice = boards.slice(0, batch);
    encodeBatch(slice);
    for (let i = 0; i < 5; i++) await evaluator.evaluate(slice);

    const runs = batch <= 8 ? 200 : 60;
    const t0 = performance.now();
    for (let i = 0; i < runs; i++) await evaluator.evaluate(slice);
    const each = (performance.now() - t0) / runs;

    log(
      `  ${String(batch).padStart(5)}   ${each.toFixed(3).padStart(7)}   ` +
        `${Math.round(batch / (each / 1000))
          .toLocaleString()
          .padStart(11)}`,
    );
  }

  // ---- a whole move ----
  log("\none move, start to finish");
  log("  sims    ms/move   ms/sim   verdict");
  for (const sims of [16, 64, 256, 800]) {
    const search = new MCTS(evaluator, { nSimulations: sims, cPuct: 1.5, fpuReduction: 0.25 });
    await search.run(boards[0]!);

    const runs = sims >= 256 ? 3 : 8;
    const t0 = performance.now();
    for (let i = 0; i < runs; i++) await search.run(boards[i % boards.length]!);
    const each = (performance.now() - t0) / runs;

    // What a player actually experiences. Past about two seconds a move stops
    // feeling like thinking and starts feeling like a hang.
    const verdict = each < 400 ? "instant" : each < 1200 ? "comfortable" : each < 2500 ? "slow" : "too slow";

    log(
      `  ${String(sims).padStart(4)}   ${each.toFixed(0).padStart(8)}   ` +
        `${(each / sims).toFixed(2).padStart(6)}   ${verdict}`,
    );
  }

  log("\ndone.");
}

main().catch((error: unknown) => {
  log(`\nFAILED: ${error instanceof Error ? error.message : String(error)}`);
});
