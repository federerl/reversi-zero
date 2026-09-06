/**
 * The hub's list of games is data the front page trusts. These pin what it must
 * hold: unique ids and paths, a playable game with rated opponents, and a
 * strongest rating that really is the maximum of what the manifest says.
 */

import { describe, expect, it } from "vitest";

import manifest from "../src/games/reversi/engine/models.json";
import { GAMES, strongestElo } from "../src/hub/registry";

describe("the game registry", () => {
  it("has unique ids and paths, each path a directory", () => {
    const ids = GAMES.map((g) => g.id);
    const paths = GAMES.map((g) => g.path);
    expect(new Set(ids).size).toBe(ids.length);
    expect(new Set(paths).size).toBe(paths.length);
    for (const path of paths) expect(path).toMatch(/^\/[a-z-]+\/$/);
  });

  it("offers Reversi as playable with every rated opponent the manifest lists", () => {
    const reversi = GAMES.find((g) => g.id === "reversi");
    expect(reversi?.status).toBe("playable");
    expect(reversi?.opponents.length).toBe(manifest.models.length + manifest.baselines.length);
  });

  it("reports the strongest rating as the manifest's maximum, and none for a planned game", () => {
    const reversi = GAMES.find((g) => g.id === "reversi")!;
    const expected = Math.max(
      ...manifest.models.map((m) => m.elo),
      ...manifest.baselines.map((b) => b.elo),
    );
    expect(strongestElo(reversi)).toBe(expected);

    const planned = GAMES.find((g) => g.status === "planned");
    expect(planned).toBeDefined();
    expect(strongestElo(planned!)).toBeUndefined();
  });
});
