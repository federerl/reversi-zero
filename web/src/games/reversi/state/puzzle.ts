/**
 * A puzzle session: the ending as it is being played out.
 *
 * The first version of this graded one move and stopped. It was not a game and
 * it did not teach: being told "+6" asks you to trust a number, where watching
 * the position resolve into a win you can count on the board shows you one. So a
 * puzzle is now played to the last square, against an opponent that cannot be
 * improved on, and **solved means won**.
 *
 * Deliberately not an extension of `game.ts`. That reducer requires a
 * `humanColor`, a `levelId` and a `modelId`, its `undo` rewinds *past the
 * agent's reply*, and its status line can say "the agent wins". Threading a mode
 * flag through all of that would leave both harder to read than the hundred
 * lines that actually differ.
 *
 * **Nothing is said about the position while the ending is in progress.** The
 * exact value is available after every move — it is a few milliseconds away —
 * and showing it would turn the page into a cheat sheet you could play by
 * watching rather than by calculating. The whole account is given at the end
 * instead, where it can name the move that lost the game.
 *
 * The opponent's replies arrive asynchronously, because the search runs in a
 * worker. That makes `thinking` a real state rather than a decoration: the board
 * must stop taking clicks while an answer is outstanding, or a fast player could
 * queue a move into a position that no longer exists.
 */

import {
  BLACK,
  apply,
  discCounts,
  isTerminal,
  legalActions,
  mustPass,
  passAction,
  type Action,
  type Player,
  type State,
} from "../engine/rules";
import type { Puzzle } from "../puzzles/data";

export interface Turn {
  readonly state: State;
  /** The move that produced this position, or null for the puzzle itself. */
  readonly move: Action | null;
  /** True when the player made it; false for the opponent. */
  readonly mine: boolean;
}

export interface Session {
  readonly stage: number;
  /** Index into the puzzles of `stage`, in the order the file lists them. */
  readonly index: number;
  /** Every position from the puzzle onwards. Never empty. */
  readonly history: readonly Turn[];
  /** True while the opponent's reply is being searched for. */
  readonly thinking: boolean;
  /** Set once the ending is over and the account has been worked out. */
  readonly outcome: Outcome | null;
  /**
   * True once the result dialog has been closed, by any route.
   *
   * Separate from `outcome` because the outcome is still wanted after the
   * dialog is gone -- it drives the move table and the stage count. Without
   * this, dismissing the dialog and then stepping through the winning line
   * would reopen it on the way back.
   */
  readonly dismissed: boolean;
  /** How many moves of the stored perfect line are shown, or null when not shown. */
  readonly line: number | null;
  /** Ids of puzzles whose ending the player has actually won. */
  readonly solvedIds: ReadonlySet<string>;
  readonly error: string | null;
  /**
   * Which attempt this is. Incremented whenever the board is reset.
   *
   * The page keys an outstanding search on the position it asked about, and two
   * runs of the same puzzle reach the same positions -- so the position alone
   * cannot tell a fresh question from one asked before the restart. Without
   * this, replaying a puzzle either reuses the previous run's answer or never
   * asks for one, and the board sits waiting on a move that will not come.
   */
  readonly run: number;
}

export interface Outcome {
  readonly discs: { readonly mine: number; readonly theirs: number };
  readonly margin: number;
  readonly won: boolean;
  /** The exact result of the player's opening move, from the shipped table. */
  readonly openingMove: Action;
  readonly openingMargin: number;
  readonly best: number;
  /**
   * The player's move that turned a win into something else, if there was one.
   * Null when they never had it, or never lost it.
   */
  readonly lostItAt: {
    /** Index of the position the move was played from. */
    readonly from: number;
    readonly move: Action;
    /** What the position was worth before it. Always positive. */
    readonly wasWorth: number;
  } | null;
}

export type SessionAction =
  | { type: "play"; action: Action }
  | { type: "thinking" }
  | { type: "opponentPlayed"; action: Action }
  | { type: "finished"; outcome: Outcome }
  | { type: "failed"; message: string }
  | { type: "dismiss" }
  | { type: "retry" }
  | { type: "showLine" }
  | { type: "stepLine" }
  | { type: "hideLine" }
  | { type: "pickStage"; stage: number }
  | { type: "pickPuzzle"; index: number }
  | { type: "next" };

export function newSession(stage: number, solvedIds: ReadonlySet<string> = new Set()): Session {
  return {
    stage,
    index: 0,
    history: [],
    thinking: false,
    outcome: null,
    dismissed: false,
    line: null,
    solvedIds,
    error: null,
    run: 0,
  };
}

/** The puzzle a session is pointing at, or null when its stage has none. */
export function currentPuzzle(session: Session, inStage: readonly Puzzle[]): Puzzle | null {
  return inStage[Math.min(session.index, inStage.length - 1)] ?? null;
}

function start(puzzle: Puzzle): Turn[] {
  return [{ state: puzzle.state, move: null, mine: true }];
}

/** The position on screen. */
export function boardOf(session: Session, puzzle: Puzzle): State {
  if (session.line !== null) {
    let state = puzzle.state;
    for (const action of puzzle.principalVariation.slice(0, session.line)) {
      state = apply(state, action);
    }
    return state;
  }
  return (session.history[session.history.length - 1] ?? { state: puzzle.state }).state;
}

/** Whose colour the player is: whoever was to move when the puzzle was set. */
export function playerColour(puzzle: Puzzle): Player {
  return puzzle.state.toMove;
}

/** True while the player may click a square. */
export function accepting(session: Session, puzzle: Puzzle): boolean {
  if (session.thinking || session.outcome !== null || session.line !== null) return false;
  const state = boardOf(session, puzzle);
  return !isTerminal(state) && state.toMove === playerColour(puzzle);
}

/**
 * The player's moves so far, each with the index of the position it was played
 * *from*.
 *
 * `from`, not the index of the turn the move produced. Those differ by one, and
 * reading one as the other is the difference between blaming the move that lost
 * a game and blaming the one after it.
 */
export function playerMoves(session: Session): Array<{ from: number; move: Action }> {
  const out: Array<{ from: number; move: Action }> = [];
  session.history.forEach((turn, index) => {
    if (turn.move !== null && turn.mine) out.push({ from: index - 1, move: turn.move });
  });
  return out;
}

export function reduce(
  session: Session,
  action: SessionAction,
  inStage: readonly Puzzle[],
): Session {
  const puzzle = currentPuzzle(session, inStage);
  const fresh = (next: Partial<Session>): Session => ({
    ...session,
    history: puzzle === null ? [] : start(puzzle),
    thinking: false,
    outcome: null,
    dismissed: false,
    line: null,
    error: null,
    run: session.run + 1,
    ...next,
  });

  switch (action.type) {
    case "play": {
      if (puzzle === null || !accepting(session, puzzle)) return session;
      const state = boardOf(session, puzzle);
      if (!legalActions(state).includes(action.action)) return session;
      // Seeds the history on the first move rather than relying on something
      // else to have done it, so a session is playable the moment it exists.
      const base = session.history.length > 0 ? session.history : start(puzzle);
      return {
        ...session,
        history: [...base, { state: apply(state, action.action), move: action.action, mine: true }],
        error: null,
      };
    }

    case "thinking":
      return { ...session, thinking: true };

    case "opponentPlayed": {
      if (puzzle === null) return session;
      const state = boardOf(session, puzzle);
      // Whose turn it is, before whether the move is legal. A search started
      // before a restart can come back after it, and a reply that is legal from
      // the puzzle position would otherwise be applied there -- as the
      // opponent's move, on the player's turn. That flips the side to move, and
      // the player finds themselves playing the colour the puzzle told them
      // they were not. Legality alone does not catch it, because the same move
      // is often legal for both sides.
      if (state.toMove === playerColour(puzzle)) return { ...session, thinking: false };
      if (!legalActions(state).includes(action.action)) {
        return { ...session, thinking: false };
      }
      return {
        ...session,
        history: [
          ...session.history,
          { state: apply(state, action.action), move: action.action, mine: false },
        ],
        thinking: false,
      };
    }

    case "finished": {
      const solvedIds = action.outcome.won
        ? new Set([...session.solvedIds, puzzle?.id ?? ""])
        : session.solvedIds;
      return { ...session, thinking: false, outcome: action.outcome, solvedIds };
    }

    case "failed":
      return { ...session, thinking: false, error: action.message };

    case "retry":
      return fresh({});

    case "dismiss":
      return { ...session, dismissed: true };

    case "showLine":
      // From the puzzle's own position. The stored line is perfect play from
      // there and opens with *a* best move, which need not be the one played.
      // Dismisses the dialog on the way: the point of watching the line is to
      // see the board, and a modal over it defeats that.
      return { ...session, line: 0, dismissed: true };

    case "stepLine": {
      if (session.line === null || puzzle === null) return session;
      return { ...session, line: Math.min(session.line + 1, puzzle.principalVariation.length) };
    }

    case "hideLine":
      return { ...session, line: null };

    case "pickStage":
      return { ...newSession(action.stage, session.solvedIds), index: 0, run: session.run + 1 };

    case "pickPuzzle": {
      const index = Math.max(0, Math.min(action.index, inStage.length - 1));
      const target = inStage[index];
      return {
        ...session,
        index,
        history: target === undefined ? [] : start(target),
        thinking: false,
        outcome: null,
        dismissed: false,
        line: null,
        error: null,
        run: session.run + 1,
      };
    }

    case "next": {
      const last = inStage.length - 1;
      // Stops at the end of a stage rather than rolling into the next. The
      // stages are a progression, and being moved into deeper positions without
      // asking is not the same as choosing to go there.
      const index = Math.min(session.index + 1, last);
      const target = inStage[index];
      return {
        ...session,
        index,
        history: target === undefined ? [] : start(target),
        thinking: false,
        outcome: null,
        dismissed: false,
        line: null,
        error: null,
        run: session.run + 1,
      };
    }
  }
}

/**
 * Turn a finished ending into the account the player is shown.
 *
 * `values[i]` is the exact value of `history[i]`, from the point of view of
 * whoever is to move there — which is what the solver returns. To read the
 * position from the *player's* side it has to be negated on the opponent's
 * turns, and getting that backwards would blame the wrong move.
 */
export function describeOutcome(
  session: Session,
  puzzle: Puzzle,
  values: readonly number[],
): Outcome {
  const colour = playerColour(puzzle);
  const final = session.history[session.history.length - 1]!.state;
  const counts = discCounts(final);
  const myDiscs = colour === BLACK ? counts.black : counts.white;
  const theirDiscs = colour === BLACK ? counts.white : counts.black;

  const fromPlayer = (index: number): number => {
    const value = values[index];
    const turn = session.history[index];
    if (value === undefined || turn === undefined) return 0;
    // The solver answers from the point of view of whoever is to move. Reading
    // it from the player's side means negating it on the opponent's turns, and
    // getting that backwards would blame the wrong move.
    return turn.state.toMove === colour ? value : -value;
  };

  let lostItAt: Outcome["lostItAt"] = null;
  for (const { from, move } of playerMoves(session)) {
    if (from < 0) continue;
    const before = fromPlayer(from);
    const after = fromPlayer(from + 1);
    if (before > 0 && after <= 0) {
      lostItAt = { from, move, wasWorth: before };
      break;
    }
  }

  const opening = session.history[1];
  const openingMove = opening?.move ?? -1;

  return {
    discs: { mine: myDiscs, theirs: theirDiscs },
    margin: myDiscs - theirDiscs,
    won: myDiscs > theirDiscs,
    openingMove,
    openingMargin: puzzle.margins.get(openingMove) ?? 0,
    best: puzzle.best,
    lostItAt,
  };
}

/** How many of a stage's endings the player has won. */
export function solvedInStage(session: Session, inStage: readonly Puzzle[]): number {
  return inStage.filter((puzzle) => session.solvedIds.has(puzzle.id)).length;
}

/** The squares a player may click at a position. */
export function playableSquares(state: State): readonly Action[] {
  const skip = passAction(state.size);
  return legalActions(state).filter((action) => action !== skip);
}

export { mustPass };
