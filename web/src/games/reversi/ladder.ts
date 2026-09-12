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
 * How hard a rung is, in one word -- chosen by its position, not by its rating.
 *
 * This used to compare the rating against fixed thresholds, and that was wrong
 * for a reason the release tournament demonstrated. Adding two entrants to that
 * round robin moved every strong network's rating by about 40 points without a
 * single one of their games being replayed: an Elo is a coordinate the fit
 * assigns given the field it measured, not a property of the network. Thresholds
 * read off one fit therefore expire the next time the field changes, and they
 * did -- two rungs 143 points apart and cleanly separated both came out
 * "Expert", while the disc-counting baseline sat four points from flipping word.
 *
 * Position is the durable fact. The ladder's *order* is measured, and the gaps
 * between neighbours are stable even when the absolute numbers are not, so the
 * nth rung of n is a claim that survives a refit. The rating itself is still
 * shown, one click away in the note, where it belongs with the interval that
 * qualifies it.
 *
 * "Strong" is deliberately not in this list, because it is already the name of a
 * thinking time. Two controls sit side by side, and a word that means one thing
 * under "Level" and another under "Thinking time" makes the panel harder to
 * read.
 */
const WORDS = ["Beginner", "Easy", "Fair", "Tough", "Expert", "Master"] as const;

function wordFor(index: number, total: number): string {
  // The vocabulary is sized for the ladder this project expects. A longer ladder
  // keeps its numbering and reuses the top word rather than inventing ranks
  // nobody asked for -- the number is the claim, the word is a handle.
  if (total <= WORDS.length) return WORDS[index] ?? WORDS[WORDS.length - 1]!;
  const scaled = Math.round((index / (total - 1)) * (WORDS.length - 1));
  return WORDS[scaled]!;
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

  const sorted = entries.sort((a, b) => a.elo - b.elo);
  return sorted.map((entry, index) => ({
    ...entry,
    level: index + 1,
    word: wordFor(index, sorted.length),
  }));
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
