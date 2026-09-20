/**
 * A puzzle session: which one is on screen, what was tried, and what happened.
 *
 * Deliberately not an extension of `game.ts`. That reducer is coupled to
 * engine play in ways that are wrong here -- it requires a `humanColor`, a
 * `levelId` and a `modelId`, its `undo` rewinds *past the agent's reply*
 * because a game has two players, and its status line can say "the agent wins".
 * A puzzle has no agent, no level and no opponent to take a move back from.
 * Threading a mode flag through all of that would make both harder to read than
 * repeating the sixty lines that matter.
 *
 * **Grading is a table lookup, not a search.** The exact result of every legal
 * move was computed before the file shipped, so a verdict is instant and
 * certain. That is the whole reason the interface can say "this loses by 2 where
 * the best move wins by 6" instead of "wrong" -- and it is why nothing here is
 * allowed to guess. If a move is somehow absent from the table, that is a broken
 * curriculum and it says so rather than inventing a margin.
 *
 * The board on screen is *derived* from the session rather than stored beside
 * it. Two positions that can disagree is the bug this avoids: the verdict and
 * the board would be describing different moves.
 */

import { apply, legalActions, passAction, type Action, type State } from "../engine/rules";
import type { Puzzle } from "../puzzles/data";

export interface Verdict {
  readonly move: Action;
  /** Exact final disc difference after this move, for the player. */
  readonly margin: number;
  /** The best difference available in this position. Always positive. */
  readonly best: number;
  readonly outcome: "wins" | "draws" | "loses";
  /** True when no move does better. Several moves can be optimal. */
  readonly optimal: boolean;
}

/** Did this answer keep the win? A draw did not: the position was winnable. */
export function solved(verdict: Verdict): boolean {
  return verdict.outcome === "wins";
}

export function grade(puzzle: Puzzle, move: Action): Verdict {
  const margin = puzzle.margins.get(move);
  if (margin === undefined) {
    // Unreachable through the board, which only offers legal squares. Loud
    // rather than a made-up verdict: a puzzle that grades a move it has no
    // answer for is worse than a puzzle that refuses to.
    throw new Error(`no exact result stored for move ${move} in puzzle ${puzzle.id}`);
  }
  return {
    move,
    margin,
    best: puzzle.best,
    outcome: margin > 0 ? "wins" : margin < 0 ? "loses" : "draws",
    optimal: margin === puzzle.best,
  };
}

export interface Session {
  readonly stage: number;
  /** Index into the puzzles of `stage`, in the order the file lists them. */
  readonly index: number;
  /** The move the player tried, or null before they have answered. */
  readonly played: Action | null;
  /**
   * How many moves of the stored winning line are on the board, or null when
   * the line is not being shown. Zero means the line is open at its start.
   */
  readonly line: number | null;
  /** Ids of puzzles answered with a winning move, this visit or a previous one. */
  readonly solvedIds: ReadonlySet<string>;
}

export type SessionAction =
  | { type: "play"; action: Action }
  | { type: "retry" }
  | { type: "showLine" }
  | { type: "stepLine" }
  | { type: "hideLine" }
  | { type: "pickStage"; stage: number }
  | { type: "pickPuzzle"; index: number }
  | { type: "next" };

export function newSession(stage: number, solvedIds: ReadonlySet<string> = new Set()): Session {
  return { stage, index: 0, played: null, line: null, solvedIds };
}

/** The puzzle a session is pointing at, or null when its stage has none. */
export function currentPuzzle(session: Session, inStage: readonly Puzzle[]): Puzzle | null {
  return inStage[Math.min(session.index, inStage.length - 1)] ?? null;
}

/**
 * The position to draw.
 *
 * Three cases, in the order they take precedence: the stored winning line is
 * being stepped through, the player has answered, or neither and the puzzle sits
 * as it was mined.
 */
export function boardOf(session: Session, puzzle: Puzzle): State {
  if (session.line !== null) {
    let state = puzzle.state;
    for (const action of puzzle.principalVariation.slice(0, session.line)) {
      state = apply(state, action);
    }
    return state;
  }
  if (session.played !== null) return apply(puzzle.state, session.played);
  return puzzle.state;
}

/** True while the player may still click a square. */
export function accepting(session: Session): boolean {
  return session.played === null && session.line === null;
}

/** The squares a player may click: the legal placements, pass excluded. */
export function playableSquares(puzzle: Puzzle): readonly Action[] {
  const skip = passAction(puzzle.state.size);
  return legalActions(puzzle.state).filter((action) => action !== skip);
}

export function reduce(
  session: Session,
  action: SessionAction,
  inStage: readonly Puzzle[],
): Session {
  switch (action.type) {
    case "play": {
      if (!accepting(session)) return session;
      const puzzle = currentPuzzle(session, inStage);
      if (puzzle === null) return session;

      const verdict = grade(puzzle, action.action);
      // A puzzle counts as solved the moment it is answered with a winning
      // move, and stays solved. Getting the next one wrong does not take it
      // back, and neither does trying again -- this is a record of what somebody
      // has worked out, not a score.
      const solvedIds = solved(verdict)
        ? new Set([...session.solvedIds, puzzle.id])
        : session.solvedIds;
      return { ...session, played: action.action, line: null, solvedIds };
    }

    case "retry":
      return { ...session, played: null, line: null };

    case "showLine":
      // From the puzzle's own position, not from the move that was played. The
      // stored line is perfect play from *here*, and it opens with a best move
      // which need not be the one the player chose.
      return { ...session, played: null, line: 0 };

    case "stepLine": {
      if (session.line === null) return session;
      const puzzle = currentPuzzle(session, inStage);
      if (puzzle === null) return session;
      return { ...session, line: Math.min(session.line + 1, puzzle.principalVariation.length) };
    }

    case "hideLine":
      return { ...session, line: null };

    case "pickStage":
      return { ...session, stage: action.stage, index: 0, played: null, line: null };

    case "pickPuzzle":
      return {
        ...session,
        index: Math.max(0, Math.min(action.index, inStage.length - 1)),
        played: null,
        line: null,
      };

    case "next": {
      // Stops at the end of a stage rather than rolling into the next one. The
      // stages are a progression, and being moved into deeper positions without
      // asking is not the same as choosing to go there.
      const last = inStage.length - 1;
      if (session.index >= last) return { ...session, played: null, line: null };
      return { ...session, index: session.index + 1, played: null, line: null };
    }
  }
}

/** How many of a stage's puzzles have been answered correctly. */
export function solvedInStage(session: Session, inStage: readonly Puzzle[]): number {
  return inStage.filter((puzzle) => session.solvedIds.has(puzzle.id)).length;
}
