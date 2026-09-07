/**
 * The hub's list of games is data the front page trusts. These pin what it must
 * hold: unique ids and paths, a playable game whose level count is the number of
 * levels that can really be played, and a strongest rating that is the maximum of
 * what the ladder says rather than a number typed into the hub.
 */

import { describe, expect, it } from "vitest";

import { LADDER } from "../src/games/reversi/ladder";
import { GAMES, strongestElo } from "../src/hub/registry";

describe("the game registry", () => {
  it("has unique ids and paths, each path a directory", () => {
    const ids = GAMES.map((g) => g.id);
    const paths = GAMES.map((g) => g.path);
    expect(new Set(ids).size).toBe(ids.length);
    expect(new Set(paths).size).toBe(paths.length);
    for (const path of paths) expect(path).toMatch(/^\/[a-z-]+\/$/);
  });

  it("counts the levels a visitor can actually play, not the rows of a report", () => {
    const reversi = GAMES.find((g) => g.id === "reversi");
    expect(reversi?.status).toBe("playable");
    expect(reversi?.levels.length).toBe(LADDER.length);
    expect(reversi?.ladderSummary).toBe("six levels, beginner to expert");
  });

  it("reports the strongest rating as the ladder's maximum, and none for a planned game", () => {
    const reversi = GAMES.find((g) => g.id === "reversi")!;
    expect(strongestElo(reversi)).toBe(Math.max(...LADDER.map((rung) => rung.elo)));

    const planned = GAMES.find((g) => g.status === "planned");
    expect(planned).toBeDefined();
    expect(strongestElo(planned!)).toBeUndefined();
    expect(planned!.ladderSummary).toBeUndefined();
  });
});
