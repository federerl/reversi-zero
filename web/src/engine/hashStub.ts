/**
 * A stand-in for the network whose answers can be reproduced exactly.
 *
 * The search fixture cannot be generated with the real network. Two runtimes
 * doing float arithmetic in a different order will not always break an exact
 * tie the same way, and one different choice early in a tree changes every
 * count after it. So the fixture uses this instead: an evaluator whose answers
 * are a plain integer hash of the position. That isolates what the fixture is
 * for -- the search arithmetic -- from what it is not, which is the network.
 *
 * This must stay in exact step with `reversi.web.fixtures` on the Python side.
 * The test vectors are asserted in both places, so a change to one without the
 * other fails immediately rather than leaving the two quietly disagreeing.
 *
 * Nothing here runs in the app. It exists so the search can be checked.
 */

import { policySize, BLACK, type State } from "./rules";
import type { Evaluator } from "./mcts";

/** Mixed into the position hash, matching `STUB_SEED_SALT` in Python. */
export const STUB_SEED_SALT = 0x9e3779b9;

/**
 * A 32-bit integer hash (the well-known "lowbias32" mixer).
 *
 * `Math.imul` rather than `*`: the multiplications overflow 32 bits, and `*`
 * would promote the result to a double and lose exactly the low bits that carry
 * the answer. `Math.imul` is defined to be a true 32-bit multiply, which is what
 * makes this reproduce the Python version bit for bit.
 */
export function mix32(x: number): number {
  x = x >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
  return (x ^ (x >>> 16)) >>> 0;
}

/**
 * A 32-bit fingerprint of a position.
 *
 * The board goes in as four 32-bit halves rather than two 64-bit numbers --
 * which is how it is already held here, and the reason this hash is expressible
 * in JavaScript at all.
 */
export function stubSeed(state: State): number {
  let seed = STUB_SEED_SALT;
  for (const part of [
    state.black.lo,
    state.black.hi,
    state.white.lo,
    state.white.hi,
    state.toMove === BLACK ? 0 : 1,
  ]) {
    seed = mix32(seed ^ part);
  }
  return seed;
}

/** One policy logit, in [-2, 2). Dividing by 2^32 is exact in both languages. */
export function stubLogit(seed: number, action: number): number {
  return (mix32(seed ^ (action + 1)) / 4294967296) * 4 - 2;
}

/** The position's score, in [-1, 1). */
export function stubValue(seed: number): number {
  return (mix32(seed ^ 0x5bf03635) / 4294967296) * 2 - 1;
}

/**
 * The evaluator the search fixture was generated with.
 *
 * `Math.fround` on every output because the Python side stores them in a float32
 * array before the search reads them. Handing the search full double precision
 * here would make this port disagree with the reference in the last few digits
 * of every prior, for no reason at all.
 */
export function hashStubEvaluator(): Evaluator {
  return {
    evaluate(states: readonly State[]) {
      const logits: Float32Array[] = [];
      const values: number[] = [];

      for (const state of states) {
        const seed = stubSeed(state);
        const width = policySize(state.size);
        const row = new Float32Array(width);
        for (let action = 0; action < width; action++) {
          row[action] = Math.fround(stubLogit(seed, action));
        }
        logits.push(row);
        values.push(Math.fround(stubValue(seed)));
      }

      return Promise.resolve({ logits, values });
    },
  };
}
