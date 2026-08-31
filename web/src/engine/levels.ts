/**
 * The two knobs a player gets, and why they are two rather than one.
 *
 * **Which generation** sets how good the intuition is. These are real
 * checkpoints from the training run, and the numbers attached to them come from
 * the cross-generation tournament -- 210 games per entrant, Bradley-Terry
 * ratings with bootstrap intervals, anchored at random play = 0. They are not
 * adjectives.
 *
 * **How many simulations** sets how hard it thinks. The same network searching
 * 800 positions plays noticeably better than one searching 16, which is the
 * whole idea behind the method: the search improves on the network's first
 * guess, and more search improves on it more.
 *
 * Keeping them separate means a beginner can play the final agent thinking
 * briefly, or an early agent thinking hard, and the difference between those two
 * is the thing this project is actually about.
 *
 * A note on the third control, the value guardrail. At the easier levels the
 * agent samples among reasonable moves rather than always taking the best one.
 * The guardrail throws away candidates whose value is much worse than the best
 * before sampling, which is what makes an easy opponent *weak* rather than
 * *stupid*: it plays a second-best move instead of giving away a corner. Filter
 * first, then sample -- the other order lets a blunder through whenever it
 * happens to be picked.
 */

export interface Level {
  readonly id: string;
  readonly label: string;
  readonly simulations: number;
  /** 0 plays the most-visited move; above 0 samples in proportion to visits. */
  readonly temperature: number;
  /** Sample among at most this many candidates, after the guardrail. */
  readonly topK?: number;
  /** Discard candidates whose value is worse than the best by more than this. */
  readonly guard: number;
  /**
   * Stop searching after this long, whatever the simulation count says.
   *
   * Measured in a browser on a 20-core laptop at four threads: 16 simulations
   * takes 55 ms, 64 takes 221 ms, 256 takes 898 ms and 800 takes 2.8 seconds.
   * The last of those is past the point where a move reads as thinking rather
   * than as a hang, and a phone would be several times slower again.
   *
   * So the deep levels are capped by time. The count becomes a ceiling and the
   * clock decides, which means the same level feels the same everywhere and
   * gets stronger on better hardware instead of slower on worse.
   */
  readonly maxMillis?: number;
  readonly description: string;
}

export const LEVELS: readonly Level[] = [
  {
    id: "casual",
    label: "Casual",
    simulations: 16,
    temperature: 0.8,
    topK: 3,
    guard: 0.35,
    description: "Looks a little way ahead and does not always take its best move.",
  },
  {
    id: "club",
    label: "Club",
    simulations: 64,
    temperature: 0.35,
    topK: 2,
    guard: 0.2,
    description: "Considers more, and mostly plays what it considers best.",
  },
  {
    id: "strong",
    label: "Strong",
    simulations: 256,
    temperature: 0,
    guard: 0.05,
    maxMillis: 1200,
    description: "Plays its best move every time. About a second per move.",
  },
  {
    id: "max",
    label: "Max",
    simulations: 800,
    temperature: 0,
    guard: 0,
    maxMillis: 2000,
    description:
      "Thinks for up to two seconds, and searches as far as your device allows in that time.",
  },
];

export function levelById(id: string): Level {
  const found = LEVELS.find((level) => level.id === id);
  if (found === undefined) {
    throw new Error(`unknown difficulty ${id}; expected one of ${LEVELS.map((l) => l.id).join(", ")}`);
  }
  return found;
}

/**
 * Choose which move to actually play, given what the search found.
 *
 * The guardrail is applied *before* sampling, never after. Sampling first and
 * then checking would mean a genuinely bad move gets played whenever it happens
 * to be drawn, which is exactly the behaviour the guardrail exists to prevent.
 */
export function chooseMove(
  result: { actions: number[]; visits: number[]; qValues: number[] },
  level: Level,
  random: () => number = Math.random,
): number {
  const { actions, visits, qValues } = result;
  if (actions.length === 0) throw new Error("no actions to choose from");
  if (actions.length === 1) return actions[0]!;

  let best = 0;
  for (let i = 1; i < visits.length; i++) if (visits[i]! > visits[best]!) best = i;
  if (level.temperature <= 0) return actions[best]!;

  // Step 1: throw away anything clearly worse than the best move available.
  const bestQ = Math.max(...qValues);
  let candidates = actions
    .map((action, index) => ({ action, visits: visits[index]!, q: qValues[index]! }))
    .filter((candidate) => bestQ - candidate.q <= level.guard);

  if (candidates.length === 0) return actions[best]!;

  // Step 2: keep the most-searched few of what survived.
  candidates.sort((a, b) => b.visits - a.visits);
  if (level.topK !== undefined) candidates = candidates.slice(0, level.topK);

  // Step 3: sample among those, sharpened by the temperature.
  const weights = candidates.map((candidate) =>
    Math.pow(Math.max(candidate.visits, 0), 1 / level.temperature),
  );
  const total = weights.reduce((a, b) => a + b, 0);
  if (!Number.isFinite(total) || total <= 0) return candidates[0]!.action;

  let draw = random() * total;
  for (let i = 0; i < candidates.length; i++) {
    draw -= weights[i]!;
    if (draw <= 0) return candidates[i]!.action;
  }
  return candidates[candidates.length - 1]!.action;
}
