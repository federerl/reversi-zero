/**
 * The games this site offers, and what the hub says about each.
 *
 * One entry per game. The hub imports a game's *ladder* -- the numbered levels a
 * player picks from -- and its board, and nothing else from it, so the hub page
 * never loads an engine. Adding a game is adding an entry here and a directory
 * under `src/games/`; the hub does not change.
 *
 * The count on the front page is the number of levels a visitor can actually
 * play, which is not the same as the number of rows in a rating report: the
 * cross-generation tournament rates two search baselines this build does not
 * offer as opponents. Counting report rows would have advertised eight levels
 * and shown six. Reading the ladder makes that impossible.
 *
 * Every number shown still comes from a rating report by way of the ladder.
 * Nothing about strength is typed here.
 */

import { LADDER, describeLadder } from "../games/reversi/ladder";

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
  /** The levels a player can choose, weakest first. Empty for a planned game. */
  readonly levels: readonly RatedEntry[];
  /** The ladder in one phrase: "six levels, beginner to expert". */
  readonly ladderSummary: string | undefined;
}

export const GAMES: readonly GameEntry[] = [
  {
    id: "reversi",
    title: "Othello",
    tagline: "Learned the game from nothing by playing itself, and runs entirely in your browser.",
    path: "/reversi/",
    status: "playable",
    levels: LADDER.map((rung) => ({ elo: rung.elo })),
    ladderSummary: describeLadder(),
  },
  {
    id: "gomoku",
    title: "Gomoku",
    tagline: "Five in a row. The same method on a different board.",
    path: "/gomoku/",
    status: "planned",
    levels: [],
    ladderSummary: undefined,
  },
];

/** The strongest measured rating a game offers, or undefined if nothing is rated. */
export function strongestElo(game: GameEntry): number | undefined {
  const rated = game.levels.map((o) => o.elo).filter((e): e is number => e !== undefined);
  return rated.length === 0 ? undefined : Math.max(...rated);
}
