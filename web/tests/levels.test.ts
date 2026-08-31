/**
 * The difficulty ladder, and the guardrail that keeps the easy end sensible.
 *
 * The levels themselves are not calibrated yet -- that needs a round robin
 * measuring them against each other, which is a separate piece of work. What is
 * checked here is the property that makes an easy opponent *fun* rather than
 * merely bad: it plays a second-best move, not a move that throws the game away.
 */

import { describe, expect, it } from "vitest";

import { LEVELS, chooseMove, levelById } from "../src/engine/levels";

describe("the ladder", () => {
  it("gets harder in one direction only", () => {
    // A ladder whose rungs are not ordered is not a ladder. Sampling gets
    // sharper and the search gets deeper at every step, together.
    for (let i = 1; i < LEVELS.length; i++) {
      expect(LEVELS[i]!.simulations).toBeGreaterThan(LEVELS[i - 1]!.simulations);
      expect(LEVELS[i]!.temperature).toBeLessThanOrEqual(LEVELS[i - 1]!.temperature);
      expect(LEVELS[i]!.guard).toBeLessThanOrEqual(LEVELS[i - 1]!.guard);
    }
  });

  it("names every level it offers", () => {
    for (const level of LEVELS) {
      expect(levelById(level.id)).toBe(level);
      expect(level.description.length).toBeGreaterThan(10);
    }
  });

  it("refuses a level it does not have", () => {
    expect(() => levelById("impossible")).toThrow(/unknown difficulty/);
  });
});

describe("choosing a move", () => {
  const strong = levelById("strong");
  const casual = levelById("casual");

  it("plays the most-searched move when the temperature is zero", () => {
    const result = { actions: [10, 20, 30], visits: [5, 40, 12], qValues: [0.1, 0.4, 0.2] };
    for (let i = 0; i < 20; i++) {
      expect(chooseMove(result, strong, () => i / 20)).toBe(20);
    }
  });

  it("never plays a move the guardrail rejected, however the dice fall", () => {
    // The point of the guardrail. Action 30 is well searched -- it would be
    // sampled often -- but its value is 0.9 below the best, far outside
    // casual's 0.35 window. It must never be played.
    const result = {
      actions: [10, 20, 30],
      visits: [30, 25, 40],
      qValues: [0.5, 0.45, -0.4],
    };

    const chosen = new Set<number>();
    for (let i = 0; i < 200; i++) {
      chosen.add(chooseMove(result, casual, () => i / 200));
    }

    expect(chosen.has(30)).toBe(false);
    expect(chosen.has(10)).toBe(true);
  });

  it("still plays something when every move is bad", () => {
    // A losing position is not a reason to return nothing. When the guardrail
    // rejects everything, fall back to the best available move.
    const result = { actions: [7], visits: [16], qValues: [-0.99] };
    expect(chooseMove(result, casual, () => 0.5)).toBe(7);
  });

  it("does actually vary at the easy end", () => {
    // The other half of the guardrail's job. If casual always played the same
    // move it would be a weaker `strong`, not a different opponent.
    const result = {
      actions: [10, 20, 30],
      visits: [30, 25, 20],
      qValues: [0.5, 0.45, 0.4],
    };

    const chosen = new Set<number>();
    for (let i = 0; i < 100; i++) chosen.add(chooseMove(result, casual, () => i / 100));

    expect(chosen.size).toBeGreaterThan(1);
  });

  it("takes the only legal move without thinking about it", () => {
    expect(chooseMove({ actions: [64], visits: [0], qValues: [0] }, casual)).toBe(64);
  });
});
