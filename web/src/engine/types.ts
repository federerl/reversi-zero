/**
 * What the interface knows about its opponent, which is deliberately very little.
 *
 * The board never learns where the agent runs. It asks an `Engine` for a move
 * and gets one back. Two things implement that interface:
 *
 *   `LocalEngine`   a Web Worker running the search on this machine
 *   `RemoteEngine`  the FastAPI server, over HTTP
 *
 * This mirrors the `Evaluator` protocol on the Python side, which is what lets
 * the same search run against a real network or a stand-in without knowing the
 * difference.
 *
 * The local one is what the site ships with, because it costs nothing to run
 * and cannot be overwhelmed. The remote one earns its place three ways: it is a
 * genuine fallback on a device where WebAssembly is blocked, it is how the
 * browser's answers get compared against the reference during development, and
 * it is the connection point if these games ever need to be played between two
 * people rather than against the agent.
 */

import type { Action, State } from "./rules";

/** What the search found, and everything the interface wants to show about it. */
export interface Thought {
  /** The move to play. */
  readonly action: Action;
  /** How the agent rates the position it just moved from, for the mover. */
  readonly value: number;
  /** Chance the mover wins, which is just the value rescaled to 0..1. */
  readonly winProbability: number;
  /** Visits per action, full width, zero on every illegal one. For the heat map. */
  readonly visits: readonly number[];
  /** Simulations actually run, which a time budget may cut short. */
  readonly simulations: number;
  /** True when the clock, not the count, ended the search. */
  readonly cutShort?: boolean;
  readonly elapsedMs: number;
  /** Which opponent answered -- the generation, not the difficulty. */
  readonly modelId: string;
}

export interface Engine {
  readonly kind: "local" | "remote";
  /** Ready to be asked for a move. Loading a network is not instant. */
  ready(): Promise<void>;
  /** Switch to a different generation of the agent. */
  useModel(modelId: string): Promise<void>;
  think(state: State, levelId: string, signal?: AbortSignal): Promise<Thought>;
  dispose(): void;
}

export class EngineError extends Error {}

// ---------------------------------------------------------------------------
// Messages across the worker boundary
// ---------------------------------------------------------------------------

export type WorkerRequest =
  | { readonly type: "load"; readonly id: number; readonly modelId: string }
  | {
      readonly type: "think";
      readonly id: number;
      readonly state: State;
      readonly levelId: string;
    }
  | { readonly type: "cancel"; readonly id: number };

export type WorkerResponse =
  | { readonly type: "loaded"; readonly id: number; readonly modelId: string }
  | { readonly type: "thought"; readonly id: number; readonly thought: Thought }
  | { readonly type: "cancelled"; readonly id: number }
  | { readonly type: "error"; readonly id: number; readonly message: string };
