/**
 * The worker side of the exact search. See `solver.ts` for why it exists.
 *
 * Errors are sent back as a message rather than thrown. A worker that throws
 * leaves the page holding a promise that never settles, which on this page would
 * look exactly like an opponent thinking forever.
 */

import { bestMove, solveExact } from "../engine/endgame";
import type { SolverRequest, SolverResponse } from "./solver";

self.addEventListener("message", (event: MessageEvent<SolverRequest>) => {
  const { id, kind, states } = event.data;
  try {
    const result =
      kind === "reply"
        ? states.map((state) => bestMove(state))
        : states.map((state) => solveExact(state));
    const response: SolverResponse = { id, result };
    self.postMessage(response);
  } catch (error) {
    const response: SolverResponse = {
      id,
      result: [],
      error: error instanceof Error ? error.message : String(error),
    };
    self.postMessage(response);
  }
});
