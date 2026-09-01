# ADR-0001: A board is two integers

**Status:** accepted, day 3 — and frozen since
**Applies to:** `game/bitboard.py`, `game/rules.py`, `game/state.py`
**Contracts:** C1 (index convention)

## The problem

The search asks "what are the legal moves here?" tens of millions of times per
training generation. Every one of those questions is answered by walking outward
from squares in eight directions, and every answer allocates a new position for
the tree to hold.

So the representation of a board is not a matter of taste. It sets the ceiling on
how many games can be played in a night, and games per night is the scarce
resource in this whole project.

## The options

**A list of lists.** `board[row][col]` holding an enum. Obviously correct, easy
to read, easy to debug. Legal-move generation is a loop over 64 squares, each
walking up to 7 steps in 8 directions.

**A flat array of 64.** Slightly better locality, same algorithm.

**Two integers, one bit per square.** One integer for black, one for white. Legal
moves are then computed for *all* squares at once, because shifting an integer
moves every disc on the board one step in the same direction simultaneously.
Eight shift-and-mask dilations replace the loop.

## The decision

**Two Python `int`s, and the obvious implementation kept alongside it forever.**

Bit `i` is square `i = row·8 + col`, row 0 at the top, column 0 at the left
(contract C1). The same numbering in the engine, the feature encoder, the
symmetry permutations, the API and the board on screen — never re-derived
anywhere.

Measured, this is 15–40× faster than the list version, with no C extension and no
build step, so it behaves identically on a laptop, in CI and on a cluster.

**`State` is frozen with `__slots__`, and `apply` returns a new one.** Copying is
normally a cost to avoid; here a board is two machine words, so a copy is cheaper
than maintaining an undo stack — and it removes the entire class of bug where a
node in the search tree quietly holds a board that something else mutated
underneath it. Those are the hardest bugs to find in a tree search, so they are
bought out at the start.

## What it costs

**Correctness stops being visible by reading the code.** `(bits & NOT_H_FILE) << 1`
is not self-evidently "move every disc one square right". That is a real loss, and
it is paid for deliberately:

`game/reference.py` is a list-of-lists implementation, written from the rules text
rather than transcribed from the fast one, and kept in the repository forever. It
is the specification. `tests/unit/test_differential.py` plays **20,000 random
games** through both engines nightly — about 1.2 million positions — and compares
them at **every move**: legal set, flip set, terminal flag, final score. Zero
mismatches. A 400-game version runs on every push.

(The plan asked for 50,000. At the measured rate of ~22 paired games per second
that is 38 minutes, which does not fit the nightly budget; 20,000 does. The
number is set by measurement and the reasoning lives in the test file.)

Writing the obvious version *first* was the other half of this: every bitboard bug
surfaced the moment it appeared, against a known-good answer, instead of being
debugged blind a day later.

### The bug this shape invites

Squares are numbered in reading order, so the square to the right of column H is
column A of the *next row* — numerically adjacent, nowhere near it on the board.
A sideways shift without care teleports discs across the edge, and the result is
a legal-looking position that is not the game.

Every sideways shift therefore drops the discs already against that edge before
shifting. That is what the `guard` mask on each direction is for, and it is the
first thing to suspect if the differential test ever fails.

## What we did not do

**numba or a C extension.** The pre-committed rule was to optimise only with a
measurement in hand. Bitboards were fast enough that `game/` never dominated a
profile, so the next step was never taken — and the repository is simpler and more
portable for it.

**Canonicalise colour into the state.** Tempting, since the network only ever sees
the mover's perspective. Rejected: the *state* stays literal — black is black —
and canonicalisation happens in the encoder. Baking it into the representation
would have made every debug print and every API payload lie about whose discs
were whose. See ADR-0002.

## Revisit if

* Profiling shows `game/*` above 50% of self-play time. It has never been close;
  the network and the tree dominate.
* A board larger than 8×8 is wanted. Above 64 squares this stops being two
  machine words and the arithmetic changes character. (The TypeScript port already
  lives with a version of this problem — JavaScript numbers are exact only to
  2⁵³, so it holds a board as two 32-bit halves. See ADR-0005.)
