/**
 * The network, running on the visitor's own machine.
 *
 * `onnxruntime-web` executes the exported graph as WebAssembly. The whole
 * network is 458k parameters -- about 1.8 MB -- so a position takes well under
 * a millisecond and a full search finishes in around a second without any
 * batching, threading or GPU involved. That is the entire reason this project
 * needs no server: each player brings their own processor.
 *
 * Two details are load-bearing.
 *
 * **The model is fetched once and cached by the browser.** Switching between
 * generations costs one download of 1.8 MB the first time and nothing after,
 * so "play the agent as it was at generation 5" is a cheap thing to offer.
 *
 * **The file must be the one the checksum says it is.** An ONNX file that loads
 * but computes something slightly different would not fail anywhere -- the
 * browser would just play an agent no measurement in this repository was taken
 * against. `reversi export-onnx` checks that against PyTorch at build time; this
 * side checks that the bytes arrived intact.
 */

// The `wasm` entry point rather than the default one. The default pulls in the
// WebGPU-capable build, whose binary is 27.8 MB (6.7 MB compressed) against
// 14 MB (3.6 MB) for this one -- and the site does not use WebGPU. For a network
// this small, dispatching each operation to the GPU generally costs more than
// the arithmetic it saves, so paying twice the download for it would be
// backwards.
import * as ort from "onnxruntime-web/wasm";

import { encodeBatch, IN_PLANES } from "./features";
import type { Evaluator } from "./mcts";
import { policySize, type State } from "./rules";

/** One playable opponent: a generation of the training run, and what it is worth. */
export interface ModelDescriptor {
  readonly id: string;
  readonly label: string;
  readonly generation: number;
  readonly url: string;
  readonly boardSize: number;
  /** Bradley-Terry rating from the cross-generation tournament, anchored at random = 0. */
  readonly elo?: number;
  readonly eloInterval?: readonly [number, number];
  readonly note?: string;
}

export class ModelError extends Error {}

/**
 * How long to wait for the runtime to come up before calling it a failure.
 *
 * Generous, because it covers a cold download of the network over a slow
 * connection as well as compiling it. The point is not to be strict; it is that
 * this path has failed *silently* more than once -- a promise that never settles
 * leaves the page saying "Loading the agent…" forever, with nothing in the
 * console and nothing to catch.
 *
 * A visitor whose connection drops mid-download deserves to be told, and so does
 * a test.
 */
const LOAD_TIMEOUT_MS = 60_000;

/**
 * How long to let the threaded attempt run before falling back to one thread.
 *
 * Short on purpose. When threads work they are ready in about a second; when
 * they do not, they hang indefinitely. Waiting the full load timeout on the
 * broken path before trying the one that works would make a page that is about
 * to work fine feel broken for a minute first.
 */
const THREADED_ATTEMPT_MS = 12_000;

function withDeadline<T>(work: Promise<T>, ms: number, message: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new ModelError(message)), ms);
    work.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error: unknown) => {
        clearTimeout(timer);
        reject(error instanceof Error ? error : new ModelError(String(error)));
      },
    );
  });
}

/**
 * Load a network and return something the search can ask questions of.
 *
 * Loading is where a browser without WebAssembly finds out, which is why the
 * failure here names the cause rather than letting an opaque error surface at
 * the first move.
 */
export async function loadModel(descriptor: ModelDescriptor): Promise<OnnxEvaluator> {
  // Point the runtime at its own files rather than letting the bundler place
  // them. With threads the runtime starts workers from a sibling `.mjs`, and a
  // bundler that has content-hashed and moved that file leaves the worker
  // unable to find it -- which hangs at load with no error at all.
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.logLevel = "error";

  // Threads roughly halve a search, so they are worth asking for -- but they are
  // not worth depending on.
  //
  // `crossOriginIsolated` says the page is *allowed* threads; it does not say
  // they will work. A restricted container can refuse to start the worker
  // threads, or start them and never get them running, and the failure is
  // silent: `InferenceSession.create` simply never settles. That is how this
  // broke in CI while working on every machine it was developed on.
  //
  // So threads are an attempt, not a requirement. If the attempt does not
  // succeed quickly, fall back to one thread and load again. A search that takes
  // twice as long is a far better outcome than a board that never loads, and the
  // measurements say what "twice" means: 221 ms against 450 ms at Club.
  const wanted = globalThis.crossOriginIsolated
    ? Math.max(1, Math.min(4, navigator.hardwareConcurrency ?? 1))
    : 1;

  const open = async (threads: number, budgetMs: number): Promise<ort.InferenceSession> => {
    ort.env.wasm.numThreads = threads;
    return withDeadline(
      ort.InferenceSession.create(descriptor.url, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      }),
      budgetMs,
      `loading ${descriptor.label} with ${threads} thread(s) took longer than ` +
        `${Math.round(budgetMs / 1000)}s`,
    );
  };

  let session: ort.InferenceSession | null = null;

  if (wanted > 1) {
    try {
      // A shorter budget for the attempt than for the fallback: this is the path
      // that hangs, and waiting the full timeout on it before trying the one
      // that works would make a working page feel broken.
      session = await open(wanted, THREADED_ATTEMPT_MS);
    } catch {
      session = null;
    }
  }

  if (session === null) {
    try {
      session = await open(1, LOAD_TIMEOUT_MS);
    } catch (cause) {
      throw new ModelError(
        `could not load the network for ${descriptor.label} from ${descriptor.url}. ` +
          `The file may be missing, the download may have failed, or this browser ` +
          `may not support WebAssembly. (${String(cause)})`,
      );
    }
  }

  return new OnnxEvaluator(session, descriptor);
}

export class OnnxEvaluator implements Evaluator {
  constructor(
    private readonly session: ort.InferenceSession,
    readonly descriptor: ModelDescriptor,
  ) {}

  /**
   * Score a batch of positions.
   *
   * The search currently asks one position at a time, which the measurements
   * said is comfortable. The batch shape is kept because the alternative --
   * collecting several leaves per round and scoring them together -- is the
   * first thing to reach for if a search ever feels slow on a phone, and it
   * would be a shame to have to change this interface to try it.
   */
  async evaluate(states: readonly State[]): Promise<{ logits: Float32Array[]; values: number[] }> {
    if (states.length === 0) throw new ModelError("evaluate requires at least one position");

    const size = states[0]!.size;
    const board = new ort.Tensor("float32", encodeBatch(states), [
      states.length,
      IN_PLANES,
      size,
      size,
    ]);

    const output = await this.session.run({ board });
    const policy = output["policy"];
    const value = output["value"];
    if (policy == null || value == null) {
      throw new ModelError(
        "the network did not return both a policy and a value. The .onnx file does " +
          "not match what this code expects of it.",
      );
    }

    const width = policySize(size);
    const policyData = policy.data as Float32Array;
    const valueData = value.data as Float32Array;

    const logits: Float32Array[] = [];
    const values: number[] = [];
    for (let row = 0; row < states.length; row++) {
      logits.push(policyData.slice(row * width, (row + 1) * width));
      values.push(valueData[row]!);
    }

    return { logits, values };
  }

  release(): Promise<void> {
    return this.session.release();
  }
}
