/**
 * The game as the interface holds it.
 *
 * One reducer, one immutable position per move, and a full history. Keeping
 * every position rather than mutating one makes taking a move back a matter of
 * dropping the last entry, and it means nothing on screen can ever be showing a
 * board that something else has quietly changed underneath it.
 *
 * The reducer knows the rules -- it is the same engine the agent is using, so
 * there is no second opinion about what is legal. What it does *not* do is
 * decide moves; that belongs to the engine, and arrives back here as an action.
 *
 * Reviewing a game is a second cursor, not a second copy. `viewing` is an index
 * into the same history; the game itself does not move. That distinction is the
 * whole reason browsing an earlier position cannot disturb a search in progress:
 * the engine is driven by the *live* position, which scrubbing never touches.
 * Taking a move back is the opposite -- it really does shorten the history, and
 * the agent is expected to notice.
 */

import {
  BLACK,
  WHITE,
  apply,
  discCounts,
  initialState,
  isTerminal,
  legalActions,
  mustPass,
  passAction,
  winner,
  type Action,
  type Player,
  type State,
} from "../engine/rules";
import type { Thought } from "../engine/types";

export const BOARD_SIZE = 8;

export interface Turn {
  readonly state: State;
  /** The move that produced this position, or null for the opening. */
  readonly move: Action | null;
  /** What the agent was thinking, when the agent was the one who moved. */
  readonly thought: Thought | null;
}

export interface Game {
  readonly history: readonly Turn[];
  /**
   * Which position is on screen: an index into `history`, or null for the live
   * one. Anything that changes the game resets it to null, because reviewing a
   * position that a new move has just invalidated is a state with no meaning.
   */
  readonly viewing: number | null;
  readonly humanColor: Player;
  readonly levelId: string;
  readonly modelId: string;
  readonly thinking: boolean;
  readonly error: string | null;
  readonly showAnalysis: boolean;
}

export type GameAction =
  | { type: "newGame"; humanColor?: Player }
  | { type: "swapSides" }
  | { type: "play"; action: Action }
  | { type: "agentPlayed"; action: Action; thought: Thought }
  | { type: "undo" }
  | { type: "view"; ply: number }
  | { type: "viewLive" }
  | { type: "thinking"; value: boolean }
  | { type: "error"; message: string | null }
  | { type: "setLevel"; levelId: string }
  | { type: "setModel"; modelId: string }
  | { type: "toggleAnalysis" };

export function newGame(humanColor: Player, levelId: string, modelId: string): Game {
  return {
    history: [{ state: initialState(BOARD_SIZE), move: null, thought: null }],
    viewing: null,
    humanColor,
    levelId,
    modelId,
    thinking: false,
    error: null,
    showAnalysis: false,
  };
}

export function current(game: Game): State {
  return game.history[game.history.length - 1]!.state;
}

export function lastTurn(game: Game): Turn {
  return game.history[game.history.length - 1]!;
}

/** How many moves have been played. The opening position is move 0. */
export function moveCount(game: Game): number {
  return game.history.length - 1;
}

/** True when the screen is showing the position the game is actually at. */
export function isLive(game: Game): boolean {
  return game.viewing === null || game.viewing >= game.history.length - 1;
}

/** Which ply is on screen: the live one unless the player is reviewing. */
export function viewedPly(game: Game): number {
  if (game.viewing === null) return game.history.length - 1;
  // Clamped rather than trusted. A stale index would otherwise reach the board
  // as `undefined` and render nothing at all.
  return Math.min(Math.max(game.viewing, 0), game.history.length - 1);
}

/** The turn on screen. Identical to `lastTurn` unless the player is reviewing. */
export function viewedTurn(game: Game): Turn {
  return game.history[viewedPly(game)]!;
}

export function isHumanTurn(game: Game): boolean {
  const state = current(game);
  return !isTerminal(state) && state.toMove === game.humanColor;
}

export function score(game: Game): { black: number; white: number } {
  return discCounts(current(game));
}

/** The win probability the agent reported, at each point it reported one. */
export function winProbabilityHistory(game: Game): Array<{ ply: number; probability: number }> {
  const points: Array<{ ply: number; probability: number }> = [];
  game.history.forEach((turn, ply) => {
    // A baseline reports no win probability, so it contributes no point rather
    // than a made-up one.
    const forAgent = turn.thought?.winProbability;
    if (forAgent === undefined) return;
    // The agent's value is from its own point of view. Flip it so the whole
    // series is read from the player's side of the board, which is the only way
    // a line going up can mean "you are doing better".
    points.push({ ply, probability: 1 - forAgent });
  });
  return points;
}

export function reduce(game: Game, action: GameAction): Game {
  switch (action.type) {
    case "newGame":
      return newGame(action.humanColor ?? game.humanColor, game.levelId, game.modelId);

    case "swapSides": {
      const swapped = game.humanColor === BLACK ? WHITE : BLACK;
      return newGame(swapped, game.levelId, game.modelId);
    }

    case "play":
    case "agentPlayed": {
      const state = current(game);
      if (!legalActions(state).includes(action.action)) {
        // Should be unreachable: the board only offers legal squares and the
        // engine asserts its own move is legal before returning it. Refusing
        // loudly beats corrupting the history.
        return { ...game, error: `move ${action.action} is not legal here` };
      }
      return {
        ...game,
        history: [
          ...game.history,
          {
            state: apply(state, action.action),
            move: action.action,
            thought: action.type === "agentPlayed" ? action.thought : null,
          },
        ],
        // A move lands on the live board, so that is what to show. Leaving the
        // view where it was would mean the player made a move and watched
        // nothing happen.
        viewing: null,
        error: null,
      };
    }

    case "undo": {
      // Step back to the last position where it was the player's turn, so one
      // "take back" undoes their move and the reply to it rather than leaving
      // the agent mid-think.
      if (game.history.length <= 1) return game;
      let history = game.history.slice(0, -1);
      while (
        history.length > 1 &&
        history[history.length - 1]!.state.toMove !== game.humanColor
      ) {
        history = history.slice(0, -1);
      }
      return { ...game, history, viewing: null, thinking: false, error: null };
    }

    case "view": {
      const last = game.history.length - 1;
      const ply = Math.min(Math.max(Math.trunc(action.ply), 0), last);
      // Landing on the final position *is* going live, rather than reviewing a
      // position that happens to be the current one. Otherwise scrubbing to the
      // end would leave the board inert with no obvious way back.
      return { ...game, viewing: ply >= last ? null : ply };
    }

    case "viewLive":
      return { ...game, viewing: null };

    case "thinking":
      return { ...game, thinking: action.value };

    case "error":
      return { ...game, error: action.message, thinking: false };

    case "setLevel":
      return { ...game, levelId: action.levelId };

    case "setModel":
      return { ...game, modelId: action.modelId };

    case "toggleAnalysis":
      return { ...game, showAnalysis: !game.showAnalysis };
  }
}

// ---------------------------------------------------------------------------

/** What to tell the player is happening, in their words rather than the code's. */
export function statusLine(game: Game): string {
  const state = current(game);

  if (isTerminal(state)) {
    const result = winner(state);
    const { black, white } = discCounts(state);
    const margin = Math.abs(black - white);
    if (result === "draw") return `A draw, ${black}–${white}.`;
    const humanWon = (result === "black" ? BLACK : WHITE) === game.humanColor;
    const line = humanWon ? "You win" : "The agent wins";
    return `${line}, ${Math.max(black, white)}–${Math.min(black, white)} (${margin} discs).`;
  }

  if (game.thinking) return "Thinking…";

  if (mustPass(state)) {
    return state.toMove === game.humanColor
      ? "You have no legal move. You must pass."
      : "The agent has no legal move and must pass.";
  }

  return state.toMove === game.humanColor ? "Your move." : "The agent is to move.";
}

export function passActionFor(state: State): Action {
  return passAction(state.size);
}
