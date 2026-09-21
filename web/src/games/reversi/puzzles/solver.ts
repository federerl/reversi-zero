/**
 * The exact search, off the main thread.
 *
 * One reply at thirteen empty squares is a few tens of milliseconds on a laptop
 * and several times that on a phone. That is short enough to feel instant and
 * long enough to drop frames if it runs where the page is drawn, so it runs in a
 * worker and the board stays live while it thinks.
 *
 * The worker is a plain module with no dependencies beyond the search itself —
 * no network, no ONNX, no model. It exists purely so a burst of integer work
 * cannot freeze a tab.
 *
 * Calls are serialised behind a single worker rather than pooled. The page only
 * ever has one question outstanding: the opponent's answer to the move just
 * played, and nothing else can be asked until that arrives.
 */

import type { Action, State } from "../engine/rules";

export interface SolverRequest {
  readonly id: number;
  readonly kind: "reply" | "values";
  readonly states: readonly State[];
}

export interface SolverResponse {
  readonly id: number;
  /** For "reply": one move. For "values": the exact value of each state. */
  readonly result: number[];
  readonly error?: string;
}

type Pending = {
  resolve: (value: number[]) => void;
  reject: (reason: Error) => void;
};

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, Pending>();

function ensureWorker(): Worker {
  if (worker !== null) return worker;
  worker = new Worker(new URL("./solver.worker.ts", import.meta.url), { type: "module" });
  worker.addEventListener("message", (event: MessageEvent<SolverResponse>) => {
    const waiting = pending.get(event.data.id);
    if (waiting === undefined) return;
    pending.delete(event.data.id);
    if (event.data.error !== undefined) waiting.reject(new Error(event.data.error));
    else waiting.resolve(event.data.result);
  });
  worker.addEventListener("error", (event) => {
    // A worker that failed to start takes every outstanding question with it.
    // Rejecting them all beats leaving the page waiting for an answer that is
    // never coming.
    const failure = new Error(event.message || "the endgame solver stopped");
    for (const [, waiting] of pending) waiting.reject(failure);
    pending.clear();
  });
  return worker;
}

function ask(kind: SolverRequest["kind"], states: readonly State[]): Promise<number[]> {
  const id = nextId++;
  const request: SolverRequest = { id, kind, states };
  return new Promise<number[]>((resolve, reject) => {
    pending.set(id, { resolve, reject });
    ensureWorker().postMessage(request);
  });
}

/** A move the opponent cannot improve on, or PASS when it has nothing to place. */
export async function replyTo(state: State): Promise<Action> {
  const [move] = await ask("reply", [state]);
  if (move === undefined) throw new Error("the solver returned no move");
  return move;
}

/**
 * The exact value of each position, from the point of view of its own mover.
 *
 * Asked once, after a game is over, to work out which move threw it away. Doing
 * it in one call rather than one per position matters only for tidiness — by
 * then every position is small.
 */
export async function valuesOf(states: readonly State[]): Promise<number[]> {
  if (states.length === 0) return [];
  return ask("values", states);
}
