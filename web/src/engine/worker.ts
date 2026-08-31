/**
 * The search, running off the main thread.
 *
 * A search is hundreds of network calls and a few hundred thousand small
 * arithmetic operations. On the main thread that would freeze the page for the
 * whole of it: no hover, no scroll, no cancel button, and a browser tab that
 * looks broken. In a worker the board stays live while the agent thinks, which
 * is also what makes "stop thinking, I changed my mind" a real action rather
 * than a decoration.
 *
 * The protocol is deliberately small -- load a model, think about a position,
 * cancel -- because every message across this boundary is a place where the two
 * sides can disagree about what state the game is in.
 */

/// <reference lib="webworker" />

import modelsManifest from "./models.json";

import { baselineById } from "./baselines";
import { levelById, chooseMove } from "./levels";
import { MCTS, searchValue, visitCounts, type SearchResult } from "./mcts";
import { loadModel, type ModelDescriptor, type OnnxEvaluator } from "./onnx";
import { isTerminal, type State } from "./rules";
import type { Thought, WorkerRequest, WorkerResponse } from "./types";

const MODELS = modelsManifest.models as unknown as ModelDescriptor[];

let evaluator: OnnxEvaluator | null = null;
let loadedId: string | null = null;

/** In-flight searches, so a cancel can actually reach one. */
const running = new Map<number, AbortController>();

function post(message: WorkerResponse): void {
  self.postMessage(message);
}

function descriptorFor(modelId: string): ModelDescriptor {
  const found = MODELS.find((model) => model.id === modelId);
  if (found === undefined) {
    throw new Error(
      `unknown opponent ${modelId}; this build offers ${MODELS.map((m) => m.id).join(", ")}`,
    );
  }
  return found;
}

async function ensureModel(modelId: string): Promise<OnnxEvaluator | null> {
  // A baseline is not a network. Nothing to fetch, nothing to compile, and no
  // reason to keep a previously loaded model alive while one is selected.
  if (baselineById(modelId) !== undefined) {
    if (evaluator !== null) {
      await evaluator.release();
      evaluator = null;
      loadedId = null;
    }
    return null;
  }

  if (evaluator !== null && loadedId === modelId) return evaluator;

  // Release the previous one first. Each session holds its own copy of the
  // weights, and keeping several alive while a player tries out generations
  // would grow without bound.
  if (evaluator !== null) {
    await evaluator.release();
    evaluator = null;
    loadedId = null;
  }

  evaluator = await loadModel(descriptorFor(modelId));
  loadedId = modelId;
  return evaluator;
}

async function think(
  state: State,
  levelId: string,
  opponentId: string,
  signal: AbortSignal,
): Promise<Thought> {
  if (isTerminal(state)) {
    throw new Error("the game is over; there is no move to make");
  }

  // A baseline answers from the rules alone. No search, so no visit counts and
  // no value -- and the difficulty setting does not apply to it, because there
  // is nothing to spend a simulation budget on.
  const baseline = baselineById(opponentId);
  if (baseline !== undefined) {
    const started = performance.now();
    return {
      action: baseline.select(state),
      visits: [],
      simulations: 0,
      elapsedMs: performance.now() - started,
      modelId: opponentId,
    };
  }

  if (evaluator === null || loadedId === null) {
    throw new Error("no opponent has been loaded yet");
  }

  const level = levelById(levelId);
  const started = performance.now();

  const search = new MCTS(evaluator, {
    nSimulations: level.simulations,
    cPuct: 1.5,
    fpuReduction: 0.25,
    ...(level.maxMillis === undefined ? {} : { maxMillis: level.maxMillis }),
  });
  const result: SearchResult = await search.run(state, signal);

  // The value is from the point of view of the player who is about to move --
  // which at this moment is the agent. The interface is told whose view it is
  // rather than left to assume, because a client that guessed "always black"
  // would show the bar backwards every other turn while looking entirely
  // plausible.
  const value = searchValue(result);

  return {
    action: chooseMove(result, level),
    value,
    winProbability: (value + 1) / 2,
    visits: visitCounts(result),
    // What it actually managed, not what it was asked for. A time budget on a
    // slow device makes those differ, and claiming the larger number would be
    // the interface lying about how hard the agent thought.
    simulations: result.simulations,
    cutShort: result.cutShort,
    elapsedMs: performance.now() - started,
    modelId: loadedId,
  };
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;

  if (request.type === "cancel") {
    running.get(request.id)?.abort();
    return;
  }

  try {
    if (request.type === "load") {
      await ensureModel(request.modelId);
      post({ type: "loaded", id: request.id, modelId: request.modelId });
      return;
    }

    const controller = new AbortController();
    running.set(request.id, controller);
    try {
      const thought = await think(
        request.state,
        request.levelId,
        request.opponentId,
        controller.signal,
      );
      if (controller.signal.aborted) post({ type: "cancelled", id: request.id });
      else post({ type: "thought", id: request.id, thought });
    } finally {
      running.delete(request.id);
    }
  } catch (cause) {
    post({
      type: "error",
      id: request.id,
      message: cause instanceof Error ? cause.message : String(cause),
    });
  }
};
