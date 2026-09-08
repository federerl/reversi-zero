/**
 * Reviewing a game.
 *
 * The property worth protecting is that reviewing is a *view*. Scrubbing back
 * must change what is drawn and nothing else -- not whose turn it is, not the
 * position the engine plays from, and above all not the identity of the search
 * in flight, because the screen keys its agent-turn effect on that and a change
 * would abort a search the player was waiting for.
 *
 * So most of these tests assert what stayed the same.
 */

import { describe, expect, it } from "vitest";

import { BLACK, legalActions, passAction } from "../src/games/reversi/engine/rules";
import type { Thought } from "../src/games/reversi/engine/types";
import {
  current,
  isHumanTurn,
  isLive,
  moveCount,
  newGame,
  reduce,
  viewedPly,
  viewedTurn,
  type Game,
} from "../src/games/reversi/state/game";

/** A game with `moves` legal moves played, alternating whoever is to move. */
function played(moves: number): Game {
  let game = newGame(BLACK, "club", "greedy");
  for (let i = 0; i < moves; i++) {
    const state = current(game);
    const legal = legalActions(state).filter((action) => action !== passAction(state.size));
    const move = legal[0];
    if (move === undefined) break;
    game = reduce(game, { type: "play", action: move });
  }
  return game;
}

function thought(action: number, probability: number): Thought {
  return {
    action,
    visits: new Array(65).fill(0),
    winProbability: probability,
    simulations: 64,
    elapsedMs: 120,
    modelId: "greedy",
  };
}

describe("reviewing a game", () => {
  it("starts live, with nothing to review", () => {
    const game = newGame(BLACK, "club", "greedy");

    expect(game.viewing).toBeNull();
    expect(isLive(game)).toBe(true);
    expect(moveCount(game)).toBe(0);
  });

  it("draws an earlier position without moving the game", () => {
    const game = played(4);
    const liveState = current(game);

    const reviewing = reduce(game, { type: "view", ply: 1 });

    expect(viewedPly(reviewing)).toBe(1);
    expect(isLive(reviewing)).toBe(false);
    expect(viewedTurn(reviewing).state).not.toBe(liveState);
    // The game itself has not moved: same history, same live position.
    expect(reviewing.history).toBe(game.history);
    expect(current(reviewing)).toBe(liveState);
  });

  it("leaves the search in flight untouched", () => {
    /**
     * This is the requirement in one test. The screen restarts the agent's
     * search when the position, the opponent or the budget changes, and it
     * identifies the position by the history length. Reviewing must not change
     * any of those, or a player who scrubs back while the AI is thinking would
     * silently cancel it.
     */
    const game = { ...played(3), thinking: true };

    const reviewing = reduce(game, { type: "view", ply: 0 });

    expect(reviewing.history.length).toBe(game.history.length);
    expect(reviewing.modelId).toBe(game.modelId);
    expect(reviewing.levelId).toBe(game.levelId);
    expect(reviewing.thinking).toBe(true);
    expect(isHumanTurn(reviewing)).toBe(isHumanTurn(game));
  });

  it("clamps a ply outside the game rather than drawing nothing", () => {
    const game = played(3);

    expect(viewedPly(reduce(game, { type: "view", ply: -5 }))).toBe(0);
    expect(viewedPly(reduce(game, { type: "view", ply: 99 }))).toBe(moveCount(game));
    expect(viewedPly({ ...game, viewing: 99 })).toBe(moveCount(game));
  });

  it("treats scrubbing to the end as going live", () => {
    /**
     * Otherwise the board would sit inert on a position that happens to be the
     * current one, with the way back off screen.
     */
    const game = played(3);

    const atEnd = reduce(game, { type: "view", ply: moveCount(game) });

    expect(atEnd.viewing).toBeNull();
    expect(isLive(atEnd)).toBe(true);
  });

  it("returns to the live position on demand", () => {
    const game = reduce(played(4), { type: "view", ply: 1 });

    expect(isLive(reduce(game, { type: "viewLive" }))).toBe(true);
  });

  it("snaps back to live when the game moves on", () => {
    /**
     * A move landing while the player is reviewing leaves them looking at a
     * position the move has just invalidated. Every action that changes the
     * history therefore clears the cursor -- a player who makes a move must see
     * it happen.
     */
    const reviewing = reduce(played(4), { type: "view", ply: 1 });
    expect(isLive(reviewing)).toBe(false);

    const state = current(reviewing);
    const move = legalActions(state).filter((a) => a !== passAction(state.size))[0]!;

    expect(isLive(reduce(reviewing, { type: "play", action: move }))).toBe(true);
    expect(isLive(reduce(reviewing, { type: "undo" }))).toBe(true);
    expect(isLive(reduce(reviewing, { type: "newGame" }))).toBe(true);
    expect(isLive(reduce(reviewing, { type: "swapSides" }))).toBe(true);
    expect(
      isLive(reduce(reviewing, { type: "agentPlayed", action: move, thought: thought(move, 0.5) })),
    ).toBe(true);
  });

  it("survives a take-back that shortens the history under the cursor", () => {
    /**
     * `undo` drops entries. A cursor left pointing past the new end would reach
     * the board as `undefined`, which renders an empty board rather than a
     * position -- so the cursor is cleared and, if one ever survived, clamped.
     */
    const game = reduce(played(6), { type: "view", ply: 5 });

    const undone = reduce(game, { type: "undo" });

    expect(undone.viewing).toBeNull();
    expect(viewedTurn(undone).state).toBe(current(undone));
    // And a stale cursor, were one to exist, still resolves to a real position.
    const stale = { ...undone, viewing: 5 };
    expect(viewedTurn(stale).state).toBe(current(undone));
  });

  it("counts moves, not positions", () => {
    /**
     * The opening board is position 0 and no moves have been played, so a game
     * with four moves has five positions. Getting this off by one puts the
     * scrubber's maximum past the end of the game.
     */
    expect(moveCount(newGame(BLACK, "club", "greedy"))).toBe(0);
    expect(moveCount(played(4))).toBe(4);
    expect(played(4).history).toHaveLength(5);
  });
});
