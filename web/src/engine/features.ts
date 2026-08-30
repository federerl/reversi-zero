/**
 * Turning a position into the numbers the network reads.
 *
 * Three square grids, always from the point of view of whoever is about to
 * move:
 *
 *     plane 0   my discs
 *     plane 1   my opponent's discs
 *     plane 2   the squares I may legally play
 *
 * The network never sees colour. Black to move with a given arrangement and
 * white to move with the mirrored arrangement are the *same problem*, so they
 * get the same input and share every bit of training signal. A "which colour am
 * I" plane would split that in half for nothing.
 *
 * The third plane is redundant -- the legal moves follow from the disc
 * positions -- but working them out means tracing runs of discs outward in
 * eight directions, which is a long chain of reasoning for a small
 * convolutional network to synthesise and it would pay for it in every
 * position. The engine has just computed the answer anyway, so it gets handed
 * over. Cheap here, expensive there.
 *
 * Contract C1 is the whole risk in this file: bit `i` becomes
 * `grid[i / size][i % size]`. Getting it wrong transposes the board, which the
 * network would read as a position that never occurs, and which no summary
 * statistic would reveal. It is checked against the Python encoder square by
 * square in `tests/rules.test.ts`.
 */

import { indices, type Bits } from "./bitboard";
import { legalPlacements, mine, theirs, type State } from "./rules";

/** Planes per position: mine, theirs, my legal placements. */
export const IN_PLANES = 3;

/**
 * One position as a flat `(3, size, size)` float array, in the layout ONNX
 * Runtime wants: plane-major, then row, then column.
 *
 * A single flat `Float32Array` rather than nested arrays, because this is
 * handed straight to the runtime as a tensor and any other shape would have to
 * be flattened on the way.
 */
export function encode(state: State): Float32Array {
  const size = state.size;
  const squares = size * size;
  const out = new Float32Array(IN_PLANES * squares);

  writePlane(out, 0 * squares, mine(state));
  writePlane(out, 1 * squares, theirs(state));
  writePlane(out, 2 * squares, legalPlacements(state));

  return out;
}

/** Several positions as one contiguous batch, ready to become a tensor. */
export function encodeBatch(states: readonly State[]): Float32Array {
  if (states.length === 0) throw new Error("encodeBatch requires at least one state");

  const size = states[0]!.size;
  const stride = IN_PLANES * size * size;
  const out = new Float32Array(states.length * stride);

  states.forEach((state, row) => {
    if (state.size !== size) {
      throw new Error("every state in a batch must have the same board size");
    }
    out.set(encode(state), row * stride);
  });

  return out;
}

function writePlane(out: Float32Array, offset: number, bits: Bits): void {
  for (const square of indices(bits)) {
    out[offset + square] = 1;
  }
}

/**
 * Read one plane back out as a bitmask.
 *
 * Only used to compare against the Python encoder, which records its planes as
 * bitmasks so the check is exact per square rather than per total.
 */
export function planeBits(encoded: Float32Array, plane: number, size: number): Bits {
  const squares = size * size;
  const offset = plane * squares;
  let lo = 0;
  let hi = 0;

  for (let square = 0; square < squares; square++) {
    if (encoded[offset + square]! > 0.5) {
      if (square < 32) lo = (lo | (1 << square)) >>> 0;
      else hi = (hi | (1 << (square - 32))) >>> 0;
    }
  }

  return { lo, hi };
}

/**
 * A softmax over the legal actions only, spread back across the full width.
 *
 * The network emits raw logits for every action including illegal ones, on
 * purpose -- it stays a plain function of its input with no rules baked in
 * (contract C5, first layer). Masking belongs to whoever knows the rules.
 *
 * The maximum is subtracted before exponentiating. Without it a large logit
 * overflows to `Infinity` and the whole distribution becomes `NaN`, which then
 * spreads silently through the search.
 */
export function maskedPolicy(logits: Float32Array | number[], legal: readonly number[]): Float32Array {
  const out = new Float32Array(logits.length);
  if (legal.length === 0) return out;

  let max = -Infinity;
  for (const action of legal) {
    const value = logits[action]!;
    if (value > max) max = value;
  }

  let total = 0;
  for (const action of legal) {
    const weight = Math.exp(logits[action]! - max);
    out[action] = weight;
    total += weight;
  }

  for (const action of legal) {
    out[action] = out[action]! / total;
  }

  return out;
}
