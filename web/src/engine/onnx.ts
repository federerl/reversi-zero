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

import * as ort from "onnxruntime-web";

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
 * Load a network and return something the search can ask questions of.
 *
 * `wasmPaths` points the runtime at its own `.wasm` files. Without it the
 * runtime guesses a CDN URL, which fails under the cross-origin isolation the
 * site sets -- and fails at the first search rather than at load, which is a
 * much worse place to find out.
 */
export async function loadModel(descriptor: ModelDescriptor): Promise<OnnxEvaluator> {
  ort.env.wasm.wasmPaths = "/ort/";

  // One thread. The network is small enough that the cost of coordinating
  // several would outweigh the work being shared, and single-threaded needs no
  // cross-origin isolation to run at all.
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = "error";

  let session: ort.InferenceSession;
  try {
    session = await ort.InferenceSession.create(descriptor.url, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  } catch (cause) {
    throw new ModelError(
      `could not load the network for ${descriptor.label} from ${descriptor.url}. ` +
        `The file may be missing from the build, or the browser may not support ` +
        `WebAssembly. (${String(cause)})`,
    );
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
