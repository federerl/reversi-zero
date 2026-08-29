# How the Reversi engine works

The engine is the code that knows the rules: where you may play, which discs
flip, when you must pass, and when the game is over. Everything else in the
project trusts it completely.

This document explains how it is built, why it is built twice, and the handful
of decisions that are easy to get wrong.

---

## Why the engine is the highest-stakes code here

A bug in the rules does not crash. Nothing turns red. The AI simply learns a
slightly different game — one where, say, a particular edge case flips the wrong
disc — and it learns that game very well. Every measurement afterwards is
meaningless, and nothing in the training loop would tell you.

Compare that to a bug in, say, the web interface: you see it immediately. Or a
bug in the training loop: the loss curve looks wrong. The rules are the one
place where being wrong is *invisible*.

So the rules are implemented twice, and the two implementations are made to
prove each other right.

---

## Two engines

| | `reference.py` | `rules.py` + `bitboard.py` |
|---|---|---|
| Board stored as | a grid of squares | two integers |
| Finding flips | walk outward in 8 directions from each square | shift the whole board 8 ways |
| Speed (random games/sec) | 39 | 369 |
| Correct by inspection? | **yes** | no |
| Ships in the training loop? | no | yes |

**`reference.py` is the specification.** It is written the most direct way
possible, so you can read it and check it against the rules of Othello. It is
never used in training.

**`rules.py` is the engine.** Roughly ten times faster, which matters because a
single training generation asks it for legal moves tens of millions of times.
But you cannot verify it by reading it.

**The test that connects them** (`tests/unit/test_differential.py`) plays random
games through both engines at once and compares them at every single move: the
legal moves, which discs each of those moves would flip, the resulting position
bit for bit, the disc counts, and the final result.

Neither engine was written from the other's code. Both were written from the
rules of Othello. So if they agree across 20,000 games — about 1.2 million
positions — the chance they are both wrong in exactly the same way is
negligible. **That agreement is the whole correctness argument.**

---

## How a board fits in two integers

An 8×8 board has 64 squares. A 64-bit integer has 64 bits. So one integer can
record "which squares hold a black disc", one bit per square:

```
square index = row × 8 + column        (row 0 at the top, column 0 on the left)
```

Two integers — one for black, one for white — describe the whole position.
Squares that are set in neither are empty. No square may be set in both, and a
test checks that after every move of every game.

That numbering appears in exactly one place per component and is never
re-derived: the engine, the neural network's input, the board rotations, the web
API, and the browser all use it. It is written down as **contract C1** in
`docs/architecture.md`.

### Why this is fast

The trick is that a bit shift moves *every disc on the board at once*.

Adding 1 to a square index moves one column right. So shifting the whole integer
left by 1 moves every disc on the board one column right, simultaneously. Adding
8 moves down a row. The eight directions are just eight different shift amounts:

| Direction | Index change | Direction | Index change |
|---|---|---|---|
| left | −1 | right | +1 |
| up | −8 | down | +8 |
| up-left | −9 | up-right | −7 |
| down-left | +7 | down-right | +9 |

So instead of looping over 64 squares and walking outward from each one, the
fast engine answers for all 64 squares at the same time.

### Finding the legal moves, in four steps

For one direction:

1. Shift my discs one step. Keep whatever landed on an opponent disc — those are
   opponent discs sitting directly next to one of mine.
2. Repeat a few times, growing each of those into the whole unbroken run of
   opponent discs behind it.
3. Shift once more. Anywhere that lands on an **empty** square is a legal move,
   because it means: empty square, then an unbroken run of opponent discs, then
   one of my discs. Which is exactly the rule.
4. Do all eight directions and combine.

### The one genuinely tricky part

Squares are numbered in reading order, so **square 7 (top-right corner) and
square 8 (start of the next row) are numerically adjacent but not adjacent on
the board.** Shifting right without care teleports a disc from the right edge to
the left edge of the row below.

The fix: before any sideways shift, delete the discs that are already against
that edge. They have nowhere legal to go, so removing them is correct, and it
stops the wraparound. That is what `Shift.guard` is in `bitboard.py`.

This is the classic bitboard bug, and it is precisely the kind of thing the
differential test catches instantly.

---

## Passing and ending the game

Reversi has a rule most board games do not: **if you cannot move, you skip your
turn.** And if *neither* player can move, the game ends — which can happen with
empty squares still on the board.

The naive implementation counts consecutive passes and ends the game at two.
That works, but it means carrying around a counter that is part of the game
state, and it invites the question "what if a pass gets recorded twice?"

We do it differently. The whole rule is three branches:

```
if I have a legal square:            play it              game continues
elif my opponent has a legal square: my only move is PASS  game continues
else:                                no moves at all       GAME OVER
```

**"Two passes in a row" is not a state this code can be in.** A second pass
could only arise when neither player can move — and that condition is caught
right here as the game being over. There is no counter, and no way to
double-count.

A full board is terminal by the same rule, not by a separate "is the board
full?" check: if there are no empty squares, neither player has a legal move.
Fewer branches, fewer places to be wrong.

This is **contract C3**.

---

## Positions never change

Playing a move returns a *new* position. The old one is untouched.

Normally you would question that — copying a whole board for every move sounds
wasteful. But a position here is two integers, so copying it costs about as much
as copying two numbers, which is what it is.

What it buys is significant. The tree search will hold on to thousands of
positions at once. If applying a move edited a position in place, a node deep in
the tree could find its board silently changed by something else. Those bugs are
among the hardest to track down in a search engine. Making positions immutable
removes the entire category.

---

## Who is "the player to move"?

Every score in this project is reported **from the point of view of whoever is
to move**, never "from black's point of view".

So `result(state)` returns `+1` if the player to move won. The same finished
board gives `+1` or `−1` depending only on whose turn it is.

That sounds like a detail. It is not. The tree search flips this sign at every
level as it passes values back up the tree, and if the convention is
inconsistent anywhere, the search confidently prefers losing moves. It does not
crash and the loss curve looks fine. One convention everywhere, no exceptions,
is the defence. This is **contract C2**.

Note also that colours are *stored* as colours — `black` is always black's
discs. The "current player's view" is produced later, in one place, where the
neural network's input is built. Keeping that conversion in a single location is
the point.

---

## The eight rotations, and free training data

Rotate a Reversi board 90°, 180°, or 270°, or mirror it across either axis or
either diagonal, and you get **eight arrangements**. The rules do not care which
one you look at — the same moves are legal, the same discs flip, the same side
wins.

That is worth a lot. Every position recorded during self-play can be shown to
the network in any of its eight orientations, with the correct answer rotated
the same way. **Eight times the training data, for the cost of shuffling 64
numbers.**

We apply a random one of the eight each time a position is used for training,
rather than storing eight copies. That saves disk space and means the network
sees the same position from a different angle each time it comes up.

### The detail that is easy to get wrong

A policy vector has **65** entries: 64 squares plus "pass". When you rotate the
board, the 64 square entries move. **The pass entry does not** — passing is not a
square, and no rotation moves it.

Rotating all 65 entries would quietly scramble every pass probability in the
training set. Nothing would crash. This is **contract C6**, and there is a test
for it.

### An honest footnote

Four of the eight rotations map the standard starting position onto itself; the
other four map it onto the mirrored start. Positions from that second group are
perfectly legal Reversi positions, but they are not reachable from the standard
opening. So augmentation shifts the training distribution very slightly.

We accept that. The correctness argument depends only on the *rules* being
symmetric, not on which positions happen to be reachable — and it is what every
AlphaZero-style Othello implementation does. Worth knowing before someone asks.

---

## What the tests actually check

| File | What it does |
|---|---|
| `tests/unit/test_rules.py` | 52 hand-written positions, each drawn as a picture you can check by eye: all eight flip directions, multi-direction flips, forced pass, game over with empty squares left, full board, wipeout, illegal moves, scoring |
| `tests/property/test_invariants.py` | Plays random games and checks, after every move, things that can never be false: no square owned by both players, "game over" means exactly "no legal actions", every legal move flips something |
| `tests/property/test_symmetry.py` | The eight rotations behave like rotations — 1,000 randomly chosen positions plus exhaustive checks on small boards |
| `tests/unit/test_differential.py` | **The main event.** Both engines play the same games and are compared at every move |

The hand-written tests find the cases someone thought of. The random ones find
the cases nobody thought of — the position 40 moves in where two rules interact
in a way that never occurred to anyone.

Both are needed. Neither alone is enough.

---

## Common questions

**"Why write the same thing twice? Isn't that wasted work?"**
About an hour of work, and it never runs during training. What it buys is a
mechanical answer to "is this bit shift correct?" instead of an argument. And
writing the fast version second means every bug shows up the moment it appears,
rather than being debugged blind a day later.

**"How do you know the engine is right?"**
Two independent implementations agree across 20,000 games, compared at every
move — roughly 1.2 million positions, with every legal move's flip set checked at
each one. Plus 52 hand-verified positions and invariant checks on random play.

**"Why bitboards instead of a 2-D array?"**
The search asks for legal moves tens of millions of times per training
generation, so the engine is on the hottest path in the project. Bitboards are
about ten times faster here, measured. The cost is that correctness stops being
obvious, which is exactly what the second engine pays for.

**"What was the hardest part?"**
Edge wraparound. Squares are numbered in reading order, so the square "right of"
the last column is the first column of the next row — numerically adjacent, but
not adjacent on the board. Every sideways shift has to drop the discs already
against that edge first.

**"What would you do differently with more time?"**
Profile before optimising further. Right now the engine is fast enough that it
is unlikely to be the bottleneck — the tree search and the neural network will
dominate. The plan is to measure on day 8 and only optimise what the measurement
points at.
