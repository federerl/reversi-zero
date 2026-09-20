/**
 * The puzzle page: the curriculum it ships, the verdict it gives, and the
 * progress it remembers.
 *
 * The meta-tests over the shipped file are the valuable half. Everything a
 * player is told on that page comes out of `puzzles.json`, and that file is
 * mined rather than regenerated in CI -- so unlike the five engine fixtures, no
 * diff is watching it. These assert the properties that make it a *curriculum*
 * rather than sixty positions: every one winnable, every one with a way to go
 * wrong, every one inside its stage's band, and the whole thing already in the
 * order the page reads it in.
 *
 * The Python side re-solves every board and checks the stored answers are true.
 * This side checks they are usable.
 */

import { describe, expect, it } from "vitest";

import puzzlesFixture from "../src/games/reversi/engine/__fixtures__/puzzles.json";
import { PUZZLES, STAGES, puzzlesInStage, stageOf } from "../src/games/reversi/puzzles/data";
import { PROGRESS_KEY, readSolved, writeSolved } from "../src/games/reversi/puzzles/progress";
import {
  accepting,
  boardOf,
  currentPuzzle,
  grade,
  newSession,
  playableSquares,
  reduce,
  solved,
  solvedInStage,
  type Session,
} from "../src/games/reversi/state/puzzle";
import {
  apply,
  discCounts,
  isTerminal,
  legalActions,
  passAction,
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

function step(session: Session, action: Parameters<typeof reduce>[1]): Session {
  return reduce(session, action, puzzlesInStage(session.stage));
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
      // Two legal moves or it is not a choice, and the table has to cover every
      // square the board will let somebody click.
      const playable = playableSquares(puzzle);
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
      const stage = stageOf(puzzle.stage);
      const [low, high] = stage.empties;
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
      const difficulties = puzzlesInStage(stage.number).map((puzzle) => puzzle.difficulty);
      expect(difficulties).toEqual([...difficulties].sort((a, b) => a - b));
    }
  });

  it("gives each puzzle a line of play that reaches the promised score", () => {
    // The replay shown after an answer is evidence, not decoration. If a stored
    // line stopped short or ended on a different score, the page would be
    // asserting a win it cannot show.
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
      // A 64-bit board is not exact as a JavaScript number. The id is built
      // from the hex strings, so a puzzle whose board arrived as a number would
      // collide with another rather than merely render wrong.
      expect(puzzle.id).toMatch(/^[0-9a-f]+:[0-9a-f]+:[01]$/);
    }
    expect(new Set(PUZZLES.map((p) => p.id)).size).toBe(PUZZLES.length);
  });
});

// ---------------------------------------------------------------------------
// Grading
// ---------------------------------------------------------------------------

describe("grading an answer", () => {
  const puzzle = PUZZLES[0]!;

  it("calls a winning move a win, and names the margin", () => {
    const move = puzzle.bestMoves[0]!;
    const verdict = grade(puzzle, move);
    expect(verdict.outcome).toBe("wins");
    expect(verdict.optimal).toBe(true);
    expect(verdict.margin).toBe(puzzle.best);
    expect(solved(verdict)).toBe(true);
  });

  it("treats a drawing move as a failure, because the position was won", () => {
    // Not pedantry. Every mined position has a win in it, so a move that only
    // draws has thrown the win away -- which is the exact mistake the page
    // exists to catch, and marking it "solved" would teach the opposite.
    const drawn = PUZZLES.flatMap((p) =>
      [...p.margins].filter(([, margin]) => margin === 0).map(([move]) => ({ p, move })),
    )[0];
    if (drawn === undefined) return; // No drawn move in this curriculum.
    const verdict = grade(drawn.p, drawn.move);
    expect(verdict.outcome).toBe("draws");
    expect(solved(verdict)).toBe(false);
  });

  it("distinguishes a win that is not the best from the best", () => {
    const suboptimal = PUZZLES.flatMap((p) =>
      [...p.margins]
        .filter(([, margin]) => margin > 0 && margin < p.best)
        .map(([move]) => ({ p, move })),
    )[0];
    if (suboptimal === undefined) return;
    const verdict = grade(suboptimal.p, suboptimal.move);
    expect(verdict.outcome).toBe("wins");
    expect(verdict.optimal).toBe(false);
    expect(solved(verdict)).toBe(true);
    expect(verdict.best).toBeGreaterThan(verdict.margin);
  });

  it("refuses a move it has no exact answer for", () => {
    // Unreachable through the board, which only offers legal squares. It throws
    // rather than inventing a margin, because a puzzle that grades a move it
    // cannot see is worse than one that refuses to.
    const illegal = [...Array(64).keys()].find((square) => !puzzle.margins.has(square))!;
    expect(() => grade(puzzle, illegal)).toThrow(/no exact result/);
  });
});

// ---------------------------------------------------------------------------
// The session
// ---------------------------------------------------------------------------

describe("working through a stage", () => {
  it("accepts one answer, then stops taking clicks", () => {
    const inStage = puzzlesInStage(firstStage);
    const puzzle = inStage[0]!;
    let session = sessionIn(firstStage);
    expect(accepting(session)).toBe(true);

    session = step(session, { type: "play", action: puzzle.bestMoves[0]! });
    expect(accepting(session)).toBe(false);

    // A second click changes nothing: the verdict on screen must stay the one
    // that was earned.
    const after = step(session, { type: "play", action: puzzle.winningMoves[0]! });
    expect(after).toBe(session);
  });

  it("records a solve and keeps it after a wrong answer on the same puzzle", () => {
    const inStage = puzzlesInStage(firstStage);
    const puzzle = inStage[0]!;
    const losing = [...puzzle.margins].find(([, margin]) => margin <= 0)?.[0];

    let session = step(sessionIn(firstStage), { type: "play", action: puzzle.bestMoves[0]! });
    expect(session.solvedIds.has(puzzle.id)).toBe(true);
    expect(solvedInStage(session, inStage)).toBe(1);

    if (losing !== undefined) {
      session = step(session, { type: "retry" });
      session = step(session, { type: "play", action: losing });
      // Still solved. This is a record of what somebody worked out, not a score
      // that a later mistake takes back.
      expect(session.solvedIds.has(puzzle.id)).toBe(true);
    }
  });

  it("shows the stored line from the puzzle, not from the move that was played", () => {
    const inStage = puzzlesInStage(firstStage);
    const puzzle = inStage[0]!;
    let session = step(sessionIn(firstStage), { type: "play", action: puzzle.bestMoves[0]! });

    session = step(session, { type: "showLine" });
    expect(session.played).toBeNull();
    expect(boardOf(session, puzzle)).toEqual(puzzle.state);

    session = step(session, { type: "stepLine" });
    expect(boardOf(session, puzzle)).toEqual(apply(puzzle.state, puzzle.principalVariation[0]!));
  });

  it("never steps the line past its end", () => {
    const puzzle = puzzlesInStage(firstStage)[0]!;
    let session = step(sessionIn(firstStage), { type: "showLine" });
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

  it("carries solved puzzles across a change of stage", () => {
    const puzzle = puzzlesInStage(firstStage)[0]!;
    let session = step(sessionIn(firstStage), { type: "play", action: puzzle.bestMoves[0]! });
    const other = STAGES[STAGES.length - 1]!.number;

    session = step(session, { type: "pickStage", stage: other });
    expect(session.stage).toBe(other);
    expect(session.index).toBe(0);
    expect(session.played).toBeNull();
    expect(session.solvedIds.has(puzzle.id)).toBe(true);
    expect(currentPuzzle(session, puzzlesInStage(other))!.stage).toBe(other);
  });

  it("clamps a puzzle chosen out of range rather than showing nothing", () => {
    const inStage = puzzlesInStage(firstStage);
    const session = step(sessionIn(firstStage), { type: "pickPuzzle", index: 999 });
    expect(session.index).toBe(inStage.length - 1);
    expect(currentPuzzle(session, inStage)).not.toBeNull();
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
    // hand. Every one of these has to read as "nothing solved" rather than
    // throwing on the way to the first render.
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
