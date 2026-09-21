/**
 * The puzzle page: the curriculum it ships, the ending it plays out, and the
 * account it gives at the end.
 *
 * The meta-tests over the shipped file are the valuable half. Everything a
 * player meets on that page comes out of `puzzles.json`, and that file is mined
 * rather than regenerated in CI -- so unlike the engine fixtures, no diff is
 * watching it. These assert the properties that make it a *curriculum* rather
 * than sixty positions: every one winnable, every one with a way to go wrong,
 * every one inside its stage's band, and the whole thing already in the order
 * the page reads it in.
 *
 * The Python side re-solves every board and checks the stored answers are true.
 * This side checks they are usable, and that an ending played against them ends
 * where they say it will.
 */

import { describe, expect, it } from "vitest";

import puzzlesFixture from "../src/games/reversi/engine/__fixtures__/puzzles.json";
import { PUZZLES, STAGES, puzzlesInStage, stageOf, type Puzzle } from "../src/games/reversi/puzzles/data";
import { PROGRESS_KEY, readSolved, writeSolved } from "../src/games/reversi/puzzles/progress";
import { bestMove, solveExact, solveRoot } from "../src/games/reversi/engine/endgame";
import {
  accepting,
  boardOf,
  currentPuzzle,
  describeOutcome,
  newSession,
  playableSquares,
  playerColour,
  reduce,
  solvedInStage,
  type Outcome,
  type Session,
  type SessionAction,
} from "../src/games/reversi/state/puzzle";
import {
  apply,
  discCounts,
  isTerminal,
  legalActions,
  passAction,
  type Action,
  type State,
} from "../src/games/reversi/engine/rules";

class FakeStorage {
  readonly data = new Map<string, string>();
  getItem(key: string) {
    return this.data.get(key) ?? null;
  }
  setItem(key: string, value: string) {
    this.data.set(key, value);
  }
}

class BrokenStorage {
  getItem(): string | null {
    throw new Error("blocked");
  }
  setItem(): void {
    throw new Error("blocked");
  }
}

const firstStage = STAGES[0]!.number;

function sessionIn(stage: number): Session {
  return newSession(stage);
}

function step(session: Session, action: SessionAction): Session {
  return reduce(session, action, puzzlesInStage(session.stage));
}

function outcomeStub(won: boolean): Outcome {
  return {
    discs: { mine: won ? 40 : 20, theirs: won ? 20 : 40 },
    margin: won ? 20 : -20,
    won,
    openingMove: 0,
    openingMargin: won ? 20 : -20,
    best: 20,
    lostItAt: null,
  };
}

// ---------------------------------------------------------------------------
// The shipped curriculum
// ---------------------------------------------------------------------------

describe("the curriculum that ships", () => {
  it("has puzzles in every stage the file declares", () => {
    expect(STAGES.length).toBeGreaterThanOrEqual(3);
    expect(PUZZLES.length).toBeGreaterThanOrEqual(STAGES.length * 5);
    for (const stage of STAGES) {
      expect(puzzlesInStage(stage.number).length).toBeGreaterThan(0);
    }
  });

  it("is winnable everywhere, and never forced", () => {
    for (const puzzle of PUZZLES) {
      expect(puzzle.best).toBeGreaterThan(0);
      expect(puzzle.winningMoves.length).toBeGreaterThan(0);
      const playable = playableSquares(puzzle.state);
      expect(playable.length).toBeGreaterThanOrEqual(2);
      expect([...puzzle.margins.keys()].sort()).toEqual([...playable].sort());
    }
  });

  it("gives every puzzle a way to go wrong", () => {
    // The selection rule, read back off the file. A position where every move
    // wins has no wrong answer; one whose losing moves are unattractive teaches
    // nothing. Either the network's own pick loses, or grabbing discs does.
    for (const puzzle of PUZZLES) {
      const losing = [...puzzle.margins.values()].filter((margin) => margin <= 0);
      expect(losing.length).toBeGreaterThan(0);
      const networkFallsForIt = (puzzle.margins.get(puzzle.tempting) ?? 1) <= 0;
      expect(networkFallsForIt || puzzle.greedyFalls).toBe(true);
    }
  });

  it("keeps every puzzle inside its stage's band of empty squares", () => {
    for (const puzzle of PUZZLES) {
      const [low, high] = stageOf(puzzle.stage).empties;
      expect(puzzle.empties).toBeGreaterThanOrEqual(low);
      expect(puzzle.empties).toBeLessThanOrEqual(high);
    }
  });

  it("puts the stages in order and never overlaps their bands", () => {
    for (let i = 1; i < STAGES.length; i += 1) {
      const earlier = STAGES[i - 1]!;
      const later = STAGES[i]!;
      expect(later.number).toBeGreaterThan(earlier.number);
      expect(later.empties[0]).toBeGreaterThan(earlier.empties[1]);
    }
  });

  it("is already in presentation order, which is what makes it a progression", () => {
    // The page reads the file top to bottom. If the generator ever stops
    // ordering it, the curriculum silently becomes a shuffle while every other
    // test here still passes.
    const stages = PUZZLES.map((puzzle) => puzzle.stage);
    expect(stages).toEqual([...stages].sort((a, b) => a - b));

    for (const stage of STAGES) {
      const difficulties = puzzlesInStage(stage.number).map((p) => p.difficulty);
      expect(difficulties).toEqual([...difficulties].sort((a, b) => a - b));
    }
  });

  it("gives each puzzle a line of play that reaches the promised score", () => {
    for (const puzzle of PUZZLES) {
      let state = puzzle.state;
      for (const action of puzzle.principalVariation) {
        expect(legalActions(state)).toContain(action);
        state = apply(state, action);
      }
      expect(isTerminal(state)).toBe(true);

      const { black, white } = discCounts(state);
      const forMover = puzzle.state.toMove === 0 ? black - white : white - black;
      expect(forMover).toBe(puzzle.best);
    }
  });

  it("identifies itself, and carries boards as hex rather than numbers", () => {
    expect(puzzlesFixture.fixture).toBe("puzzles");
    expect(puzzlesFixture.pass_action).toBe(passAction(puzzlesFixture.board_size));
    for (const puzzle of PUZZLES) {
      // A 64-bit board is not exact as a JavaScript number. The id is built from
      // the hex strings, so a puzzle whose board arrived as a number would
      // collide with another rather than merely render wrong.
      expect(puzzle.id).toMatch(/^[0-9a-f]+:[0-9a-f]+:[01]$/);
    }
    expect(new Set(PUZZLES.map((p) => p.id)).size).toBe(PUZZLES.length);
  });
});

// ---------------------------------------------------------------------------
// Playing the ending out
// ---------------------------------------------------------------------------

describe("playing a puzzle out", () => {
  it("starts with the puzzle position and the player on move", () => {
    const puzzle = puzzlesInStage(firstStage)[0]!;
    const session = sessionIn(firstStage);
    expect(boardOf(session, puzzle)).toEqual(puzzle.state);
    expect(playerColour(puzzle)).toBe(puzzle.state.toMove);
    expect(accepting(session, puzzle)).toBe(true);
    expect(currentPuzzle(session, puzzlesInStage(firstStage))).toBe(puzzle);
  });

  it("takes a move, then refuses clicks until the reply arrives", () => {
    const puzzle = puzzlesInStage(firstStage)[0]!;
    const move = playableSquares(puzzle.state)[0]!;

    let session = step(sessionIn(firstStage), { type: "play", action: move });
    expect(session.history).toHaveLength(2);
    expect(session.history[1]!.mine).toBe(true);

    // It is the opponent's turn now, and it must keep refusing while the search
    // is outstanding, or a fast player could queue a move into a position that
    // no longer exists.
    expect(accepting(session, puzzle)).toBe(false);
    session = step(session, { type: "thinking" });
    expect(accepting(session, puzzle)).toBe(false);
  });

  it("refuses a move that is not legal in the position on screen", () => {
    const puzzle = puzzlesInStage(firstStage)[0]!;
    const illegal = [...Array(64).keys()].find((sq) => !puzzle.margins.has(sq))!;
    const session = step(sessionIn(firstStage), { type: "play", action: illegal });
    expect(session.history).toHaveLength(0);
  });

  it("plays a whole ending against perfect replies and reaches a finish", () => {
    const puzzle = puzzlesInStage(firstStage)[0]!;
    let session = sessionIn(firstStage);

    for (let guard = 0; guard < 80; guard += 1) {
      const state = boardOf(session, puzzle);
      if (isTerminal(state)) break;
      if (state.toMove === playerColour(puzzle)) {
        const options = playableSquares(state);
        const move = options.length > 0 ? options[0]! : passAction(state.size);
        session = step(session, { type: "play", action: move });
      } else {
        session = step(session, { type: "opponentPlayed", action: bestMove(state) });
      }
    }

    expect(isTerminal(boardOf(session, puzzle))).toBe(true);
  });

  it("counts a puzzle solved only when the ending is actually won", () => {
    // The decision this page turns on. Finding the right first move and then
    // losing the ending is not solving it.
    const puzzle = puzzlesInStage(firstStage)[0]!;
    const session = sessionIn(firstStage);

    expect(step(session, { type: "finished", outcome: outcomeStub(false) }).solvedIds.has(puzzle.id)).toBe(false);
    expect(step(session, { type: "finished", outcome: outcomeStub(true) }).solvedIds.has(puzzle.id)).toBe(true);
  });

  it("starts again from the puzzle, keeping what was already won", () => {
    const puzzle = puzzlesInStage(firstStage)[0]!;
    let session = step(sessionIn(firstStage), { type: "finished", outcome: outcomeStub(true) });
    session = step(session, { type: "play", action: playableSquares(puzzle.state)[0]! });

    session = step(session, { type: "retry" });
    expect(session.history).toHaveLength(1);
    expect(session.outcome).toBeNull();
    expect(boardOf(session, puzzle)).toEqual(puzzle.state);
    expect(session.solvedIds.has(puzzle.id)).toBe(true);
    expect(solvedInStage(session, puzzlesInStage(firstStage))).toBe(1);
  });

  it("shows the stored line from the puzzle, not from the moves played", () => {
    const puzzle = puzzlesInStage(firstStage)[0]!;
    let session = step(sessionIn(firstStage), {
      type: "play",
      action: playableSquares(puzzle.state)[0]!,
    });

    session = step(session, { type: "showLine" });
    expect(boardOf(session, puzzle)).toEqual(puzzle.state);
    expect(accepting(session, puzzle)).toBe(false);

    session = step(session, { type: "stepLine" });
    expect(boardOf(session, puzzle)).toEqual(apply(puzzle.state, puzzle.principalVariation[0]!));

    for (let i = 0; i < puzzle.principalVariation.length + 5; i += 1) {
      session = step(session, { type: "stepLine" });
    }
    expect(session.line).toBe(puzzle.principalVariation.length);
    expect(isTerminal(boardOf(session, puzzle))).toBe(true);
  });

  it("stops at the end of a stage instead of rolling into the next one", () => {
    const inStage = puzzlesInStage(firstStage);
    let session = sessionIn(firstStage);
    for (let i = 0; i < inStage.length + 3; i += 1) session = step(session, { type: "next" });
    expect(session.index).toBe(inStage.length - 1);
    expect(session.stage).toBe(firstStage);
  });

  it("carries won puzzles across a change of stage", () => {
    const puzzle = puzzlesInStage(firstStage)[0]!;
    let session = step(sessionIn(firstStage), { type: "finished", outcome: outcomeStub(true) });
    const other = STAGES[STAGES.length - 1]!.number;

    session = step(session, { type: "pickStage", stage: other });
    expect(session.stage).toBe(other);
    expect(session.index).toBe(0);
    expect(session.outcome).toBeNull();
    expect(session.solvedIds.has(puzzle.id)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// The account given at the end
// ---------------------------------------------------------------------------

describe("explaining how it ended", () => {
  function playOut(puzzle: Puzzle, choose: (state: State, ply: number) => Action): Session {
    const inStage = puzzlesInStage(puzzle.stage);
    let session = newSession(puzzle.stage);
    session = { ...session, index: inStage.indexOf(puzzle) };

    for (let ply = 0; ply < 80; ply += 1) {
      const state = boardOf(session, puzzle);
      if (isTerminal(state)) break;
      if (state.toMove === playerColour(puzzle)) {
        const options = playableSquares(state);
        const move = options.length === 0 ? passAction(state.size) : choose(state, ply);
        session = reduce(session, { type: "play", action: move }, inStage);
      } else {
        session = reduce(session, { type: "opponentPlayed", action: bestMove(state) }, inStage);
      }
    }
    return session;
  }

  function account(puzzle: Puzzle, session: Session): Outcome {
    return describeOutcome(
      session,
      puzzle,
      session.history.map((turn) => solveExact(turn.state)),
    );
  }

  it("reports a win when the player held it, and blames no move", () => {
    // Perfect play on both sides: the promised score has to appear on the board.
    // If it did not, the page would be promising something it cannot produce.
    const puzzle = PUZZLES[0]!;
    const outcome = account(puzzle, playOut(puzzle, (state) => bestMove(state)));

    expect(outcome.won).toBe(true);
    expect(outcome.margin).toBe(puzzle.best);
    expect(outcome.discs.mine - outcome.discs.theirs).toBe(puzzle.best);
    expect(outcome.lostItAt).toBeNull();
  });

  it("blames the opening move when the opening move was the mistake", () => {
    const puzzle = PUZZLES[0]!;
    const losing = [...puzzle.margins].find(([, margin]) => margin <= 0)![0];
    const outcome = account(puzzle, playOut(puzzle, (state, ply) => (ply === 0 ? losing : bestMove(state))));

    expect(outcome.openingMove).toBe(losing);
    expect(outcome.openingMargin).toBeLessThanOrEqual(0);
    expect(outcome.won).toBe(false);
    // Nothing later can be blamed: the win was gone before the second move.
    expect(outcome.lostItAt?.from ?? 0).toBe(0);
  });

  it("names the later move that threw a won game away", () => {
    // Open correctly, then play the worst move available at the next turn. The
    // account has to point at *that* move, not at the opening -- which is the
    // whole reason the exact value of every position along the line is computed.
    const puzzle = PUZZLES.find((p) => p.empties >= 6) ?? PUZZLES[0]!;
    let blundered = false;

    const session = playOut(puzzle, (state, ply) => {
      if (ply > 0 && !blundered && playableSquares(state).length > 1) {
        blundered = true;
        return [...solveRoot(state)].sort((a, b) => a[1] - b[1])[0]![0];
      }
      return bestMove(state);
    });

    const outcome = account(puzzle, session);
    expect(outcome.openingMargin).toBeGreaterThan(0);
    expect(blundered).toBe(true);
    if (!outcome.won) {
      expect(outcome.lostItAt).not.toBeNull();
      expect(outcome.lostItAt!.from).toBeGreaterThan(0);
      expect(outcome.lostItAt!.wasWorth).toBeGreaterThan(0);
    }
  });
});

// ---------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------

describe("remembering which puzzles are solved", () => {
  it("round-trips through storage", () => {
    const storage = new FakeStorage();
    writeSolved(storage, new Set(["b:w:0", "a:c:1"]));
    expect(readSolved(storage)).toEqual(new Set(["a:c:1", "b:w:0"]));
    // Sorted on the way out, so saving the same set twice does not rewrite it
    // in a different order.
    expect(storage.data.get(PROGRESS_KEY)).toBe(JSON.stringify(["a:c:1", "b:w:0"]));
  });

  it("reads nothing as nothing solved", () => {
    expect(readSolved(new FakeStorage())).toEqual(new Set());
    expect(readSolved(null)).toEqual(new Set());
    expect(readSolved(undefined)).toEqual(new Set());
  });

  it("survives a corrupt or hostile value", () => {
    // A private window, a half-written value, or somebody who edited the key by
    // hand. Every one has to read as "nothing solved" rather than throwing on
    // the way to the first render.
    const storage = new FakeStorage();
    for (const value of ["", "{", "null", '"solved"', "42", '{"a":1}']) {
      storage.data.set(PROGRESS_KEY, value);
      expect(readSolved(storage)).toEqual(new Set());
    }
    storage.data.set(PROGRESS_KEY, '["real", 7, null, {"x":1}]');
    expect(readSolved(storage)).toEqual(new Set(["real"]));
  });

  it("survives storage that throws on every call", () => {
    const broken = new BrokenStorage();
    expect(readSolved(broken)).toEqual(new Set());
    expect(() => writeSolved(broken, new Set(["a"]))).not.toThrow();
  });
});
