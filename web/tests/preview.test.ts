/**
 * The launcher's board shows a real position.
 *
 * It is replayed from a recorded self-play game, so this pins two things: that
 * the replay is legal all the way (`apply` refuses anything else), and that the
 * position is far enough in to look like a game rather than an opening.
 */

import { describe, expect, it } from "vitest";

import games from "../src/games/reversi/engine/__fixtures__/games.json";
import { discCounts, initialState, legalActions } from "../src/games/reversi/engine/rules";
import { PREVIEW_PLIES, previewPosition } from "../src/hub/preview";

describe("the launcher's preview position", () => {
  it("is a legal replay of a game the agent actually played", () => {
    const moves = (games as { games: Array<{ moves: number[] }> }).games[0]!.moves;
    expect(moves.length).toBeGreaterThan(PREVIEW_PLIES);
    // previewPosition throws if any move in the replay is illegal.
    expect(() => previewPosition()).not.toThrow();
  });

  it("shows a game in progress, not an opening or an ending", () => {
    const state = previewPosition();
    const { black, white } = discCounts(state);
    const opening = discCounts(initialState(8));

    expect(black + white).toBe(opening.black + opening.white + PREVIEW_PLIES);
    expect(black).toBeGreaterThan(2);
    expect(white).toBeGreaterThan(2);
    expect(legalActions(state).length).toBeGreaterThan(0);
  });

  it("replays exactly as many plies as it says", () => {
    const four = discCounts(previewPosition(4));
    expect(four.black + four.white).toBe(8);
  });
});
