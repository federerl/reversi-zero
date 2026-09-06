/**
 * The two opponents that are not the neural agent.
 *
 * The trained network is strong even at its earliest checkpoint: generation 5
 * rates +547, which is *above* the hand-written depth-4 alpha-beta search it was
 * measured against. Turning its simulation budget down does not make it a
 * beginner's opponent, because the strength is in the network's own opinion and
 * the search only sharpens that. There was no rung on the ladder below +547, so
 * a new player had nothing to beat.
 *
 * These fill it in, and they are not invented for the purpose: they are the
 * baselines the whole project was measured against, with ratings from the same
 * round robin as every generation.
 *
 *     random      0 Elo   the anchor everything else is measured from
 *     greedy   +313 Elo   takes the most discs available this move
 *     gen 5    +547 Elo   six hours of self-play
 *     gen 60   +877 Elo   the finished agent
 *
 * They also cost nothing to offer. Neither needs the 1.8 MB network, so picking
 * one is instant, and a visitor who only ever plays Random never downloads a
 * model at all.
 *
 * Ported from `reversi.agents.random_agent` and `reversi.agents.greedy`.
 */

import { popcount } from "./bitboard";
import { flips, legalActions, passAction, type Action, type State } from "./rules";

export interface Baseline {
  readonly id: string;
  readonly label: string;
  /** Which entrant in the tournament report this is, for its rating. */
  readonly ratingName: string;
  readonly note: string;
  select(state: State, random?: () => number): Action;
}

/**
 * Uniformly at random over the legal moves.
 *
 * The anchor for every rating in this project: 0 Elo by definition, because Elo
 * only measures differences and something has to be the zero.
 */
export const RANDOM_BASELINE: Baseline = {
  id: "random",
  label: "Random",
  ratingName: "random",
  note: "Plays a legal move chosen at random. The zero that every rating here is measured from.",
  select(state, random = Math.random) {
    const actions = legalActions(state);
    if (actions.length === 0) {
      throw new Error("asked for a move in a finished position");
    }
    return actions[Math.floor(random() * actions.length)]!;
  },
};

/**
 * Takes whichever move flips the most discs right now.
 *
 * A natural first idea, and a bad strategy: Reversi rewards *restricting the
 * opponent's options*, and taking discs early usually does the opposite. It is
 * a good opponent to learn against precisely because beating it is the moment
 * that idea clicks.
 *
 * Ties are broken at random rather than by board order. Always taking the
 * lowest-numbered square would make this prefer the top-left corner, which is a
 * positional bias rather than a greedy one -- and it would quietly flatter any
 * agent that learned to exploit it.
 */
export const GREEDY_BASELINE: Baseline = {
  id: "greedy",
  label: "Greedy",
  ratingName: "greedy",
  note: "Always takes the most discs available this move — which is a worse idea than it sounds.",
  select(state, random = Math.random) {
    const actions = legalActions(state);
    if (actions.length === 0) {
      throw new Error("asked for a move in a finished position");
    }
    if (actions.length === 1 && actions[0] === passAction(state.size)) return actions[0]!;

    const gains = actions.map((action) => popcount(flips(state, action)));
    const best = Math.max(...gains);
    const tied = actions.filter((_, index) => gains[index] === best);
    return tied[Math.floor(random() * tied.length)]!;
  },
};

export const BASELINES: readonly Baseline[] = [RANDOM_BASELINE, GREEDY_BASELINE];

export function baselineById(id: string): Baseline | undefined {
  return BASELINES.find((baseline) => baseline.id === id);
}

export function isBaseline(id: string): boolean {
  return baselineById(id) !== undefined;
}
