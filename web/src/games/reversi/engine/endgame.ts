/**
 * An exact endgame search, in the browser.
 *
 * This is the third implementation of the same idea in this project: the Python
 * solver in `reversi.endgame`, the reference engine it was checked against, and
 * this. It exists so the puzzle page can *play an ending out* — you make a move,
 * the opponent answers with a move that cannot be improved on, and the score you
 * were promised actually appears on the board.
 *
 * Precomputing that was not an option. The opponent's replies are forced, but
 * yours are not, so a stored tree branches on every move you make — roughly
 * fifteen thousand positions per puzzle at the deepest stage, times sixty
 * puzzles. A search is smaller than its own answer here.
 *
 * **It is fast enough, and that was measured before it was written.** Ranking
 * every move at thirteen empty squares takes Python about ten seconds; the
 * single search this page actually needs — one reply to one move — takes about
 * forty milliseconds here, because tight integer bit-twiddling is the thing
 * JavaScript does far better than CPython. Every move after the first is
 * cheaper, since the board only fills up.
 *
 * ## Why this file does not use `Bits`
 *
 * The rest of the engine holds a board as `{lo, hi}`, which is immutable, easy
 * to reason about and allocates. That is the right trade at one position per
 * move and the wrong one at a hundred thousand positions per search: the
 * allocation alone would dominate. Here a board is four plain numbers passed as
 * four arguments, and results come back through module-level scratch variables
 * rather than in objects.
 *
 * That is a deliberate, local exception, and it is why this file re-derives
 * shifting and move generation instead of importing them. The duplication is
 * the cost; the guard against it is that a **generated fixture holds this search
 * to the Python solver's answers**, position by position and move by move, the
 * same way `rules.ts` is held to the Python rules.
 *
 * ## Scoring
 *
 * Returns the plain disc difference, giving empty squares to nobody — contract
 * C3, and what every published number in this project uses. Some Othello tooling
 * awards leftover empties to the winner; the two agree whenever perfect play
 * fills the board, which is most of the time, and that near-agreement is exactly
 * what makes the difference dangerous rather than obvious.
 */

import { BLACK, passAction, type Action, type State } from "./rules";

const SIZE = 8;
const FULL = 0xffffffff >>> 0;

/** A board is `(lo, hi)`: squares 0–31 in `lo`, 32–63 in `hi`. */
interface Direction {
  readonly delta: number;
  readonly guardLo: number;
  readonly guardHi: number;
}

const DIRECTIONS: readonly Direction[] = buildDirections();

function buildDirections(): Direction[] {
  let notLastLo = FULL;
  let notLastHi = FULL;
  let notFirstLo = FULL;
  let notFirstHi = FULL;

  for (let row = 0; row < SIZE; row++) {
    const last = row * SIZE + SIZE - 1;
    const first = row * SIZE;
    if (last < 32) notLastLo = (notLastLo & ~(1 << last)) >>> 0;
    else notLastHi = (notLastHi & ~(1 << (last - 32))) >>> 0;
    if (first < 32) notFirstLo = (notFirstLo & ~(1 << first)) >>> 0;
    else notFirstHi = (notFirstHi & ~(1 << (first - 32))) >>> 0;
  }

  // The same eight directions, in the same order, as the Python engine.
  const steps: ReadonlyArray<readonly [number, number]> = [
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, -1],
    [0, 1],
    [1, -1],
    [1, 0],
    [1, 1],
  ];

  return steps.map(([dRow, dCol]) => ({
    delta: dRow * SIZE + dCol,
    // A disc already against the edge it would leave is cleared before the
    // shift, rather than being teleported to the far side of the board.
    guardLo: dCol > 0 ? notLastLo : dCol < 0 ? notFirstLo : FULL,
    guardHi: dCol > 0 ? notLastHi : dCol < 0 ? notFirstHi : FULL,
  }));
}

// Scratch registers. Ugly, and the point: a search that allocated a pair of
// numbers per shift would spend most of its time in the garbage collector.
let shiftLo = 0;
let shiftHi = 0;
let movesLo = 0;
let movesHi = 0;
let flipLo = 0;
let flipHi = 0;

function shiftBy(lo: number, hi: number, dir: Direction): void {
  const gLo = (lo & dir.guardLo) >>> 0;
  const gHi = (hi & dir.guardHi) >>> 0;
  const delta = dir.delta;
  if (delta > 0) {
    shiftLo = (gLo << delta) >>> 0;
    shiftHi = ((gHi << delta) | (gLo >>> (32 - delta))) >>> 0;
  } else {
    const n = -delta;
    shiftLo = ((gLo >>> n) | (gHi << (32 - n))) >>> 0;
    shiftHi = (gHi >>> n) >>> 0;
  }
}

function popcount32(x: number): number {
  let v = x >>> 0;
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  v = (v + (v >>> 4)) & 0x0f0f0f0f;
  return (Math.imul(v, 0x01010101) >>> 24) & 0xff;
}

function popcount(lo: number, hi: number): number {
  return popcount32(lo) + popcount32(hi);
}

/** Squares where the mover may place, into `movesLo` / `movesHi`. */
function placements(
  mineLo: number,
  mineHi: number,
  theirsLo: number,
  theirsHi: number,
): void {
  const emptyLo = ~(mineLo | theirsLo) >>> 0;
  const emptyHi = ~(mineHi | theirsHi) >>> 0;
  let outLo = 0;
  let outHi = 0;

  for (const dir of DIRECTIONS) {
    shiftBy(mineLo, mineHi, dir);
    let runLo = (shiftLo & theirsLo) >>> 0;
    let runHi = (shiftHi & theirsHi) >>> 0;
    let cursorLo = runLo;
    let cursorHi = runHi;

    while (cursorLo !== 0 || cursorHi !== 0) {
      shiftBy(cursorLo, cursorHi, dir);
      cursorLo = (shiftLo & theirsLo) >>> 0;
      cursorHi = (shiftHi & theirsHi) >>> 0;
      runLo = (runLo | cursorLo) >>> 0;
      runHi = (runHi | cursorHi) >>> 0;
    }

    shiftBy(runLo, runHi, dir);
    outLo = (outLo | (shiftLo & emptyLo)) >>> 0;
    outHi = (outHi | (shiftHi & emptyHi)) >>> 0;
  }

  movesLo = outLo;
  movesHi = outHi;
}

/** The discs `square` would turn over, into `flipLo` / `flipHi`. */
function flipsFrom(
  square: number,
  mineLo: number,
  mineHi: number,
  theirsLo: number,
  theirsHi: number,
): void {
  const seedLo = square < 32 ? (1 << square) >>> 0 : 0;
  const seedHi = square < 32 ? 0 : (1 << (square - 32)) >>> 0;
  let outLo = 0;
  let outHi = 0;

  for (const dir of DIRECTIONS) {
    let runLo = 0;
    let runHi = 0;
    shiftBy(seedLo, seedHi, dir);
    let cursorLo = (shiftLo & theirsLo) >>> 0;
    let cursorHi = (shiftHi & theirsHi) >>> 0;

    while (cursorLo !== 0 || cursorHi !== 0) {
      runLo = (runLo | cursorLo) >>> 0;
      runHi = (runHi | cursorHi) >>> 0;
      shiftBy(cursorLo, cursorHi, dir);
      cursorLo = (shiftLo & theirsLo) >>> 0;
      cursorHi = (shiftHi & theirsHi) >>> 0;
    }

    // The run only flips if one of my own discs closes it.
    shiftBy(runLo, runHi, dir);
    if ((shiftLo & mineLo) >>> 0 !== 0 || (shiftHi & mineHi) >>> 0 !== 0) {
      outLo = (outLo | runLo) >>> 0;
      outHi = (outHi | runHi) >>> 0;
    }
  }

  flipLo = outLo;
  flipHi = outHi;
}

/**
 * Squares from most to least promising, by shape alone.
 *
 * A corner can never be flipped once taken, so it is the one square whose value
 * does not depend on the rest of the position; the squares beside a corner are
 * the worst for the mirror-image reason. A tiebreak, not an evaluation — the
 * search still computes the true value of every move. Its only job is to try
 * likely-good moves first, so alpha-beta can discard the rest unexamined.
 */
const SHAPE_RANK: readonly number[] = (() => {
  const scored: Array<[number, number]> = [];
  for (let row = 0; row < SIZE; row++) {
    for (let col = 0; col < SIZE; col++) {
      const nearRow = Math.min(row, SIZE - 1 - row);
      const nearCol = Math.min(col, SIZE - 1 - col);
      let rank: number;
      if (nearRow === 0 && nearCol === 0) rank = 0;
      else if (nearRow <= 1 && nearCol <= 1) rank = 4;
      else if (nearRow === 0 || nearCol === 0) rank = 1;
      else if (nearRow === 1 || nearCol === 1) rank = 3;
      else rank = 2;
      scored.push([rank, row * SIZE + col]);
    }
  }
  scored.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const place = new Array<number>(SIZE * SIZE);
  scored.forEach(([, square], index) => {
    place[square] = index;
  });
  return place;
})();

/**
 * Empty-square count above which ordering by opponent mobility pays for itself.
 * Deeper in the tree it buys far more in cut-offs than the extra move generation
 * costs; near the leaves the tree is small enough that it does not.
 */
const MOBILITY_ORDERING_FROM = 9;

const EXACT = 0;
const LOWER = 1;
const UPPER = 2;

/**
 * Negate a score from the child's point of view to the parent's.
 *
 * `| 0` is not decoration. Plain `-x` turns a drawn position's 0 into `-0`,
 * which compares equal to 0 everywhere and is *not* equal to it under
 * `Object.is` — so a draw would leak out of this module as a value that looks
 * right, prints as `-0`, and fails an exact comparison against the Python
 * solver's 0. Coercing through int32 removes the sign.
 */
function negate(value: number): number {
  return -value | 0;
}

function squaresOf(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < 32; i++) if (((lo >>> i) & 1) !== 0) out.push(i);
  for (let i = 0; i < 32; i++) if (((hi >>> i) & 1) !== 0) out.push(i + 32);
  return out;
}

/**
 * Negamax with alpha-beta. Returns the final disc difference for the mover.
 *
 * `alpha` and `beta` bracket the values still worth knowing: at or below alpha
 * the caller already has better, at or above beta the caller will avoid this
 * branch entirely. Inside the window the value is exact; outside it is only a
 * bound, which is why the table records which of the two it holds.
 *
 * The table never expires an entry. A search that stops early would have to
 * record "worth X if you look six more plies"; this one always runs to the end,
 * so a position's value is a property of the position alone.
 */
function search(
  mineLo: number,
  mineHi: number,
  theirsLo: number,
  theirsHi: number,
  alpha: number,
  beta: number,
  table: Map<string, number>,
): number {
  const key = `${mineLo},${mineHi},${theirsLo},${theirsHi}`;
  const cached = table.get(key);
  if (cached !== undefined) {
    const value = cached >> 2;
    const flag = cached & 3;
    if (flag === EXACT) return value;
    if (flag === LOWER && value >= beta) return value;
    if (flag === UPPER && value <= alpha) return value;
  }

  placements(mineLo, mineHi, theirsLo, theirsHi);
  const myLo = movesLo;
  const myHi = movesHi;

  if (myLo === 0 && myHi === 0) {
    placements(theirsLo, theirsHi, mineLo, mineHi);
    if (movesLo === 0 && movesHi === 0) {
      // Contract C3: neither side can place, so the game is over.
      return popcount(mineLo, mineHi) - popcount(theirsLo, theirsHi);
    }
    // A pass is not a move and is never stored: it is the same position seen
    // from the other side.
    return negate(
      search(theirsLo, theirsHi, mineLo, mineHi, -beta, -alpha, table),
    );
  }

  const moves = squaresOf(myLo, myHi);
  if (moves.length > 1) {
    const remaining =
      64 - popcount((mineLo | theirsLo) >>> 0, (mineHi | theirsHi) >>> 0);
    if (remaining < MOBILITY_ORDERING_FROM) {
      moves.sort((a, b) => SHAPE_RANK[a]! - SHAPE_RANK[b]!);
    } else {
      const replies = new Map<number, number>();
      for (const square of moves) {
        flipsFrom(square, mineLo, mineHi, theirsLo, theirsHi);
        const bitLo = square < 32 ? (1 << square) >>> 0 : 0;
        const bitHi = square < 32 ? 0 : (1 << (square - 32)) >>> 0;
        placements(
          (theirsLo & ~flipLo) >>> 0,
          (theirsHi & ~flipHi) >>> 0,
          (mineLo | flipLo | bitLo) >>> 0,
          (mineHi | flipHi | bitHi) >>> 0,
        );
        replies.set(square, popcount(movesLo, movesHi));
      }
      moves.sort((a, b) => replies.get(a)! - replies.get(b)! || a - b);
    }
  }

  const originalAlpha = alpha;
  let best = -(SIZE * SIZE + 1);
  let window = alpha;

  for (const square of moves) {
    flipsFrom(square, mineLo, mineHi, theirsLo, theirsHi);
    const bitLo = square < 32 ? (1 << square) >>> 0 : 0;
    const bitHi = square < 32 ? 0 : (1 << (square - 32)) >>> 0;
    const value = negate(
      search(
        (theirsLo & ~flipLo) >>> 0,
        (theirsHi & ~flipHi) >>> 0,
        (mineLo | flipLo | bitLo) >>> 0,
        (mineHi | flipHi | bitHi) >>> 0,
        -beta,
        -window,
        table,
      ),
    );
    if (value > best) best = value;
    if (best > window) window = best;
    if (window >= beta) break;
  }

  const flag = best <= originalAlpha ? UPPER : best >= beta ? LOWER : EXACT;
  table.set(key, (best << 2) | flag);
  return best;
}

// ---------------------------------------------------------------------------
// The public surface
// ---------------------------------------------------------------------------

/**
 * The most empty squares this will accept.
 *
 * Not a property of the algorithm — it would start on thirty and never return.
 * It is where the measured cost stops being something a page can wait for: the
 * cost roughly doubles and a half per extra square, and the deepest stage this
 * project ships is thirteen. Refusing above the limit turns a mistake in a
 * caller into an error instead of a frozen tab.
 */
export const MAX_EMPTIES = 14;

export class TooManyEmptiesError extends Error {}

function halves(state: State): [number, number, number, number] {
  const black = state.black;
  const white = state.white;
  return state.toMove === BLACK
    ? [black.lo, black.hi, white.lo, white.hi]
    : [white.lo, white.hi, black.lo, black.hi];
}

export function emptyCount(state: State): number {
  return (
    64 -
    popcount(
      (state.black.lo | state.white.lo) >>> 0,
      (state.black.hi | state.white.hi) >>> 0,
    )
  );
}

function guard(state: State): void {
  const open = emptyCount(state);
  if (open > MAX_EMPTIES) {
    throw new TooManyEmptiesError(
      `this position has ${open} empty squares and the limit is ${MAX_EMPTIES}`,
    );
  }
}

/**
 * The final disc difference from the point of view of the player to move.
 *
 * Positive means the mover wins by that many discs with perfect play on both
 * sides; negative means they lose by that many.
 */
export function solveExact(state: State): number {
  guard(state);
  const [mineLo, mineHi, theirsLo, theirsHi] = halves(state);
  return search(mineLo, mineHi, theirsLo, theirsHi, -65, 65, new Map()) | 0;
}

/** Exact final disc difference after each legal move, for the mover. */
export function solveRoot(state: State): Map<Action, number> {
  guard(state);
  const [mineLo, mineHi, theirsLo, theirsHi] = halves(state);
  placements(mineLo, mineHi, theirsLo, theirsHi);

  // Each root move gets a full window, so its value comes back exact rather than
  // as a bound meaning only "no better than the one before". The shared table
  // makes that far cheaper than it sounds.
  const table = new Map<string, number>();
  const margins = new Map<Action, number>();

  for (const square of squaresOf(movesLo, movesHi)) {
    flipsFrom(square, mineLo, mineHi, theirsLo, theirsHi);
    const bitLo = square < 32 ? (1 << square) >>> 0 : 0;
    const bitHi = square < 32 ? 0 : (1 << (square - 32)) >>> 0;
    margins.set(
      square,
      negate(
        search(
          (theirsLo & ~flipLo) >>> 0,
          (theirsHi & ~flipHi) >>> 0,
          (mineLo | flipLo | bitLo) >>> 0,
          (mineHi | flipHi | bitHi) >>> 0,
          -65,
          65,
          table,
        ),
      ),
    );
  }
  return margins;
}

/**
 * A move that cannot be improved on, or PASS when there is nothing to place.
 *
 * Ties are broken by the shape ordering rather than arbitrarily, so the same
 * position always draws the same reply. An opponent that answered differently
 * on a retry would make the puzzle unrepeatable, which is worse than one that is
 * predictable.
 */
export function bestMove(state: State): Action {
  guard(state);
  const [mineLo, mineHi, theirsLo, theirsHi] = halves(state);
  placements(mineLo, mineHi, theirsLo, theirsHi);
  if (movesLo === 0 && movesHi === 0) return passAction(state.size);

  const table = new Map<string, number>();
  let chosen = -1;
  let bestValue = -65;

  // Alpha-beta *between* root moves, unlike `solveRoot`. That function has to
  // report an exact value for every move, so each gets a full window; this one
  // only has to name a best move, so once a move worth +6 is in hand the rest
  // need only be asked "can you beat +6?", which is a far cheaper question. At
  // thirteen empty squares it is the difference between a page that pauses and
  // one that does not.
  for (const square of squaresOf(movesLo, movesHi).sort(
    (a, b) => SHAPE_RANK[a]! - SHAPE_RANK[b]!,
  )) {
    flipsFrom(square, mineLo, mineHi, theirsLo, theirsHi);
    const bitLo = square < 32 ? (1 << square) >>> 0 : 0;
    const bitHi = square < 32 ? 0 : (1 << (square - 32)) >>> 0;
    const value = negate(
      search(
        (theirsLo & ~flipLo) >>> 0,
        (theirsHi & ~flipHi) >>> 0,
        (mineLo | flipLo | bitLo) >>> 0,
        (mineHi | flipHi | bitHi) >>> 0,
        -65,
        -bestValue,
        table,
      ),
    );
    if (value > bestValue) {
      bestValue = value;
      chosen = square;
    }
  }
  return chosen;
}
