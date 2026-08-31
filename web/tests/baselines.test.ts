/**
 * The two opponents that are not the neural agent.
 *
 * They exist because the network is strong even at its earliest checkpoint, so
 * there was no rung on the ladder a new player could beat. What matters is that
 * they are the *same* Random and Greedy the tournament measured -- otherwise the
 * ratings shown beside them in the interface describe something else.
 */

import { describe, expect, it } from "vitest";

import gamesFixture from "../src/engine/__fixtures__/games.json";
import manifest from "../src/engine/models.json";

import { fromHex, popcount } from "../src/engine/bitboard";
import { BASELINES, GREEDY_BASELINE, RANDOM_BASELINE, baselineById } from "../src/engine/baselines";
import {
  BLACK,
  WHITE,
  apply,
  flips,
  initialState,
  isTerminal,
  legalActions,
  type Player,
  type State,
} from "../src/engine/rules";

const SIZE = 8;

function stateOf(blackHex: string, whiteHex: string, toMove: number): State {
  return {
    black: fromHex(blackHex),
    white: fromHex(whiteHex),
    toMove: (toMove === 0 ? BLACK : WHITE) as Player,
    size: SIZE,
  };
}

/** Every position from the recorded games, so these see real boards. */
function positions(limit: number): State[] {
  const out: State[] = [];
  for (const game of gamesFixture.games) {
    for (const row of game.positions) {
      const [black, white, toMove] = row as [string, string, number];
      const state = stateOf(black, white, toMove);
      if (!isTerminal(state)) out.push(state);
      if (out.length >= limit) return out;
    }
  }
  return out;
}

describe("both baselines", () => {
  it("only ever play a legal move", () => {
    // The property that matters most. A baseline that could produce an illegal
    // move would corrupt a game rather than lose one.
    const boards = positions(400);
    expect(boards.length).toBeGreaterThan(300);

    for (const baseline of BASELINES) {
      for (const state of boards) {
        const legal = legalActions(state);
        for (let draw = 0; draw < 4; draw++) {
          const chosen = baseline.select(state, () => draw / 4);
          expect(legal).toContain(chosen);
        }
      }
    }
  });

  it("can play a whole game to the end without getting stuck", () => {
    for (const baseline of BASELINES) {
      let state = initialState(SIZE);
      let plies = 0;
      while (!isTerminal(state) && plies < 200) {
        state = apply(state, baseline.select(state));
        plies++;
      }
      expect(isTerminal(state)).toBe(true);
      expect(plies).toBeGreaterThan(30);
    }
  });

  it("refuses to move in a finished position rather than inventing one", () => {
    const finished: State = { black: fromHex("0"), white: fromHex("ffffffffffffffff"), toMove: BLACK, size: SIZE };
    for (const baseline of BASELINES) {
      expect(() => baseline.select(finished)).toThrow(/finished position/);
    }
  });

  it("is addressable by the id the interface uses", () => {
    for (const baseline of BASELINES) {
      expect(baselineById(baseline.id)).toBe(baseline);
    }
    expect(baselineById("gen60")).toBeUndefined();
  });
});

describe("greedy", () => {
  it("always takes the most discs available", () => {
    // This is its whole definition, and it is what the +313 rating measured.
    for (const state of positions(300)) {
      const actions = legalActions(state);
      const best = Math.max(...actions.map((a) => popcount(flips(state, a))));
      for (let draw = 0; draw < 3; draw++) {
        const chosen = GREEDY_BASELINE.select(state, () => draw / 3);
        expect(popcount(flips(state, chosen))).toBe(best);
      }
    }
  });

  it("breaks ties without preferring the top-left of the board", () => {
    // Always taking the lowest-numbered square would be a positional bias rather
    // than a greedy one, and it would quietly flatter any agent that learned to
    // exploit it. From the opening every move flips exactly one disc, so all
    // four are tied and the choice must depend only on the draw.
    const start = initialState(SIZE);
    const chosen = new Set([0, 0.3, 0.55, 0.9].map((d) => GREEDY_BASELINE.select(start, () => d)));
    expect(chosen.size).toBeGreaterThan(1);
  });
});

describe("random", () => {
  it("can reach every legal move", () => {
    const start = initialState(SIZE);
    const legal = legalActions(start);
    const seen = new Set<number>();
    for (let i = 0; i < 100; i++) seen.add(RANDOM_BASELINE.select(start, () => i / 100));
    expect([...seen].sort((a, b) => a - b)).toEqual(legal);
  });
});

describe("the ladder they complete", () => {
  it("puts a beatable opponent below the weakest network", () => {
    // The reason these exist. Generation 5 rates +547, above the depth-4 search
    // it was measured against, so before this the easiest opponent on offer was
    // already stronger than a classical engine.
    const weakestNetwork = Math.min(...manifest.models.map((m) => m.elo));

    for (const baseline of BASELINES) {
      const rated = manifest.baselines.find((b) => b.name === baseline.ratingName);
      expect(rated, `${baseline.id} has no measured rating`).toBeDefined();
      expect(rated!.elo).toBeLessThan(weakestNetwork);
    }
  });

  it("labels every opponent with a rating from the same tournament", () => {
    // The standing rule: labels state measured strength, never adjectives. A
    // baseline with no entry in the report would appear unlabelled, which is
    // worse than not offering it.
    for (const baseline of BASELINES) {
      const rated = manifest.baselines.find((b) => b.name === baseline.ratingName);
      expect(rated!.elo).toBeGreaterThanOrEqual(0);
      expect(baseline.note.length).toBeGreaterThan(20);
    }
  });
});
