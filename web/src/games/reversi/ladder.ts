/**
 * The difficulty ladder a player actually sees: Level 1 to Level 6.
 *
 * A player wants a rung, not a checkpoint. "Generation 20" is meaningless
 * before you know what generations are, and "+758 Elo" is meaningless before
 * you know what the scale is anchored to. So the interface offers numbered
 * levels with a word attached, and every technical fact stays one click away.
 *
 * **The ladder is derived, not typed.** The rungs are the opponents this build
 * can actually play, ordered by the rating each of them measured in the
 * cross-generation tournament. Nothing here decides that generation 20 is
 * harder than generation 5; the tournament did, and this reads the result. Add a
 * generation to the manifest and a rung appears in the right place with no code
 * change.
 *
 * **One scale only.** Every number used here comes from the cross-generation
 * table, where all six played each other and random play is 0. The difficulty
 * calibration is a separate tournament with its own anchor, and mixing the two
 * would order the ladder by numbers that were never compared -- which is exactly
 * the mistake `docs/experiments.md` warns about. The calibration's numbers are
 * still shown, but only against the thinking time they belong to.
 *
 * The word beside a level comes from the rating band, so it says something true
 * about strength rather than being a name invented for a rung.
 */

import { BASELINES } from "./engine/baselines";
import modelsManifest from "./engine/models.json";

export interface Rung {
  /** 1-based position on the ladder. What the player picks. */
  readonly level: number;
  /** One word for how hard it is, from the measured rating. */
  readonly word: string;
  /** The opponent this rung plays as -- a baseline id, or a generation id. */
  readonly modelId: string;
  /** What it is, for the disclosure: "Generation 20", "Greedy". */
  readonly opponentLabel: string;
  /** Measured rating in the cross-generation tournament, random play = 0. */
  readonly elo: number;
  readonly interval: readonly [number, number] | undefined;
  /** What this opponent does, in plain words. From the manifest or the baseline. */
  readonly note: string | undefined;
  /** True when a neural network plays this rung, so a search budget applies. */
  readonly usesNetwork: boolean;
}

interface ManifestModel {
  id: string;
  label: string;
  elo?: number;
  eloInterval?: [number, number];
  note?: string;
}

interface ManifestBaseline {
  name: string;
  elo: number;
  interval: [number, number];
}

/**
 * How hard a rating feels, in one word.
 *
 * The thresholds are read off the ladder the tournament produced: random play at
 * 0, a disc-counting heuristic at +313, the earliest checkpoint at +547, and the
 * finished agent at +877.
 *
 * The top two rungs share a word on purpose. Generations 40 and 60 are 22 rating
 * points apart with intervals that overlap heavily, so calling one of them
 * "strong" and the other "expert" would invent a difference the tournament did
 * not find. They are both expert; the numbers behind them are in the note.
 *
 * "Strong" is deliberately not one of these words, because it is already the
 * name of a thinking time. Two controls sit side by side, and a word that means
 * one thing under "Level" and another under "Thinking time" is a word that makes
 * the panel harder to read.
 */
function wordFor(elo: number): string {
  if (elo < 150) return "Beginner";
  if (elo < 400) return "Easy";
  if (elo < 600) return "Fair";
  if (elo < 750) return "Tough";
  return "Expert";
}

function buildLadder(): Rung[] {
  const models = modelsManifest.models as unknown as ManifestModel[];
  const rated = modelsManifest.baselines as unknown as ManifestBaseline[];

  const entries = [
    ...BASELINES.map((baseline) => {
      const rating = rated.find((entry) => entry.name === baseline.ratingName);
      return {
        modelId: baseline.id,
        opponentLabel: baseline.label,
        elo: rating?.elo ?? 0,
        interval: rating ? ([rating.interval[0], rating.interval[1]] as const) : undefined,
        note: baseline.note,
        usesNetwork: false,
      };
    }),
    ...models.map((model) => ({
      modelId: model.id,
      opponentLabel: model.label,
      elo: model.elo ?? 0,
      interval: model.eloInterval
        ? ([model.eloInterval[0], model.eloInterval[1]] as const)
        : undefined,
      note: model.note,
      usesNetwork: true,
    })),
  ];

  return entries
    .sort((a, b) => a.elo - b.elo)
    .map((entry, index) => ({ ...entry, level: index + 1, word: wordFor(entry.elo) }));
}

/** Every rung, weakest first. */
export const LADDER: readonly Rung[] = buildLadder();

/** The rung an opponent id belongs to. Falls back to the first, so the UI always has one. */
export function rungFor(modelId: string): Rung {
  return LADDER.find((rung) => rung.modelId === modelId) ?? LADDER[0]!;
}

/** "Level 4", the name a player recognises. */
export function rungName(rung: Rung): string {
  return `Level ${rung.level}`;
}

/** How the ladder is described in one phrase: "six levels, beginner to expert". */
export function describeLadder(): string {
  const words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
  const count = words[LADDER.length] ?? String(LADDER.length);
  const first = LADDER[0]?.word.toLowerCase() ?? "";
  const last = LADDER[LADDER.length - 1]?.word.toLowerCase() ?? "";
  return `${count} levels, ${first} to ${last}`;
}
