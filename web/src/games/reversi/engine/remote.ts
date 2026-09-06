/**
 * The opponent, running on a server. The fallback, not the default.
 *
 * This talks to the FastAPI service in `src/reversi/api/`, which is already
 * built and tested. It exists for three reasons, none of which is "the site
 * needs it":
 *
 *  - **A device where WebAssembly does not work.** Rare, but not zero: some
 *    corporate browser policies disable it outright. Falling back to a server
 *    is better than a blank board and an apology.
 *  - **Checking the browser against the reference.** Two implementations of the
 *    same search are only useful if they can be pointed at the same position
 *    and compared, which needs both to be reachable from one place.
 *  - **The path to playing another person.** That is the one feature that
 *    genuinely cannot be done without a server, and this is where it would
 *    attach.
 *
 * Boards cross the wire as 16-digit hex strings, never as JSON numbers. A
 * 64-bit board does not survive `Number` -- the top eleven squares round away,
 * nothing errors, and the two sides simply start disagreeing about the position.
 */

import { fromHex, toHex } from "./bitboard";
import { BLACK, WHITE, type Player, type State } from "./rules";
import { EngineError, type Engine, type Thought } from "./types";

interface WirePosition {
  black: string;
  white: string;
  to_move: "black" | "white";
  board_size: number;
}

export function toWire(state: State): WirePosition {
  return {
    black: toHex(state.black),
    white: toHex(state.white),
    to_move: state.toMove === BLACK ? "black" : "white",
    board_size: state.size,
  };
}

export function fromWire(position: WirePosition): State {
  return {
    black: fromHex(position.black),
    white: fromHex(position.white),
    toMove: (position.to_move === "black" ? BLACK : WHITE) as Player,
    size: position.board_size,
  };
}

export class RemoteEngine implements Engine {
  readonly kind = "remote" as const;

  constructor(
    private readonly baseUrl = "",
    private modelId = "server",
  ) {}

  async ready(): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/health`);
    if (!response.ok) {
      throw new EngineError(
        `the server is not answering (${response.status}). It may not be running.`,
      );
    }
  }

  useModel(modelId: string): Promise<void> {
    // The server serves whichever checkpoint it was started with. Choosing a
    // generation is a local-engine feature; saying so plainly beats pretending
    // the request worked and quietly playing a different opponent.
    this.modelId = modelId;
    return Promise.resolve();
  }

  async think(state: State, levelId: string, signal?: AbortSignal): Promise<Thought> {
    const started = performance.now();

    const response = await fetch(`${this.baseUrl}/api/ai-move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        position: toWire(state),
        difficulty: levelId,
        want_analysis: true,
      }),
      ...(signal ? { signal } : {}),
    });

    if (response.status === 429) {
      throw new EngineError("the server is busy right now. Try again in a moment.");
    }
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new EngineError(
        detail?.detail?.error ?? `the server refused the move (${response.status})`,
      );
    }

    const body = await response.json();
    const value: number = body.evaluation.value;

    return {
      action: body.action,
      value,
      winProbability: body.evaluation.win_probability,
      visits: body.analysis?.visits ?? [],
      simulations: body.analysis?.simulations ?? 0,
      elapsedMs: body.analysis?.elapsed_ms ?? performance.now() - started,
      modelId: this.modelId,
    };
  }

  dispose(): void {
    // Nothing to release: the server holds no session, which is the whole point
    // of its being stateless.
  }
}
