/**
 * The ladder a player picks from.
 *
 * It exists to hide checkpoint names and ratings behind numbered levels, so the
 * thing to guard is that hiding them does not make it lie: the order has to be
 * the order the tournament measured, every rung has to be an opponent this build
 * can actually play, and the numbers have to come from the manifest rather than
 * from here.
 */

import { describe, expect, it } from "vitest";

import { BASELINES } from "../src/games/reversi/engine/baselines";
import { LEVELS } from "../src/games/reversi/engine/levels";
import manifest from "../src/games/reversi/engine/models.json";
import { LADDER, describeLadder, rungFor, rungName } from "../src/games/reversi/ladder";

describe("the difficulty ladder", () => {
  it("offers exactly the opponents this build can play", () => {
    expect(LADDER).toHaveLength(BASELINES.length + manifest.models.length);
    const ids = LADDER.map((rung) => rung.modelId);
    for (const baseline of BASELINES) expect(ids).toContain(baseline.id);
    for (const model of manifest.models) expect(ids).toContain(model.id);
  });

  it("is numbered from 1 and ordered by measured rating, weakest first", () => {
    expect(LADDER.map((rung) => rung.level)).toEqual(LADDER.map((_, i) => i + 1));
    const ratings = LADDER.map((rung) => rung.elo);
    expect(ratings).toEqual([...ratings].sort((a, b) => a - b));
    expect(ratings[0]).toBe(0); // random play anchors the scale
  });

  it("takes every rating from the manifest, never from itself", () => {
    for (const model of manifest.models) {
      const rung = rungFor(model.id);
      expect(rung.elo).toBe(model.elo);
      expect(rung.interval).toEqual(model.eloInterval);
      expect(rung.opponentLabel).toBe(model.label);
      expect(rung.usesNetwork).toBe(true);
    }
    for (const baseline of manifest.baselines) {
      const rung = LADDER.find((entry) => entry.modelId === baseline.name);
      if (rung === undefined) continue; // rated in the table but not offered to play
      expect(rung.elo).toBe(baseline.elo);
      expect(rung.usesNetwork).toBe(false);
    }
  });

  it("gives a harder word to a higher rating, and never a weaker one", () => {
    const words = LADDER.map((rung) => rung.word);
    const rank = ["Beginner", "Easy", "Fair", "Tough", "Expert"];
    const positions = words.map((word) => rank.indexOf(word));
    expect(positions).not.toContain(-1);
    expect(positions).toEqual([...positions].sort((a, b) => a - b));

    // A level word must not repeat a thinking-time label. The two controls are
    // read together, so a word cannot mean two different things.
    for (const level of LEVELS) expect(words).not.toContain(level.label);
  });

  it("names a rung the way the interface does, and the ladder as a phrase", () => {
    expect(rungName(LADDER[0]!)).toBe("Level 1");
    expect(rungName(LADDER[LADDER.length - 1]!)).toBe(`Level ${LADDER.length}`);
    expect(describeLadder()).toMatch(/^\w+ levels, \w+ to \w+$/);
  });

  it("always resolves an opponent id to a rung", () => {
    expect(rungFor("gen60").opponentLabel).toBe("Generation 60");
    expect(rungFor("nothing-like-this")).toBe(LADDER[0]);
  });
});
