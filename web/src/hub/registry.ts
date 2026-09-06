/**
 * The games this site offers, and what the hub says about each.
 *
 * One entry per game. The hub imports a game's *manifest* -- the generated list
 * of rated opponents -- and nothing else from it, so the hub page stays a few
 * kilobytes and never loads an engine. Adding a game is adding an entry here and
 * a directory under `src/games/`; the hub does not change.
 *
 * Every number shown on a card comes from the manifest, which is generated from
 * a rating report. The rule that a difficulty label states a measured strength
 * and never an adjective applies to the hub too.
 */

import reversiManifest from "../games/reversi/engine/models.json";

export interface RatedEntry {
  readonly elo?: number;
}

export interface GameEntry {
  readonly id: string;
  readonly title: string;
  /** One sentence, said plainly, about what this agent is. */
  readonly tagline: string;
  /** Where the game lives. A directory with its own HTML file, see vite.config.ts. */
  readonly path: string;
  readonly status: "playable" | "planned";
  /** Rated opponents the game offers: the networks plus the fixed baselines. */
  readonly opponents: readonly RatedEntry[];
}

const reversiOpponents: RatedEntry[] = [
  ...reversiManifest.baselines.map((baseline) => ({ elo: baseline.elo })),
  ...reversiManifest.models.map((model) => ({ elo: model.elo })),
];

export const GAMES: readonly GameEntry[] = [
  {
    id: "reversi",
    title: "Reversi",
    tagline: "Learned the game from nothing by playing itself. Runs entirely in your browser.",
    path: "/reversi/",
    status: "playable",
    opponents: reversiOpponents,
  },
  {
    id: "gomoku",
    title: "Gomoku",
    tagline: "Five in a row. The same method, a different board.",
    path: "/gomoku/",
    status: "planned",
    opponents: [],
  },
];

/** The strongest measured rating a game offers, or undefined if nothing is rated. */
export function strongestElo(game: GameEntry): number | undefined {
  const rated = game.opponents.map((o) => o.elo).filter((e): e is number => e !== undefined);
  return rated.length === 0 ? undefined : Math.max(...rated);
}
