# The endgame: solved rather than estimated

Every number this project publishes carries an error bar, because every number is
measured. A rating comes from a tournament, a win rate from a few hundred games,
a calibration from a few thousand. The honest thing to attach to each of them is
an interval.

Near the end of an Othello game that is the wrong instrument. With a dozen empty
squares the position is small enough to search to the very last move, so the
answer can be a **proof** instead of an estimate. This is the one part of the
repository whose claims have no interval, and it is worth being precise about why
it is allowed to.

Three pieces, in order: a solver, a rule for what makes a good puzzle, and a page
that asks them.

---

## Why an estimate is the wrong tool here

The agent's opinion about who is winning comes from a neural network, improved by
a tree search that looks at a few hundred positions and stops. It has to stop:
from the opening position there are more Othello games than atoms worth counting,
so any search must cut itself off somewhere and ask the network to guess the rest.
The guess is good and it is still a guess.

Two facts make that unsatisfying at the end of a game.

**The tree stops being big.** A position with ten empty squares has, after
alpha-beta pruning, on the order of a hundred thousand positions below it. That
is seconds of work, not years. The reason to guess has gone away.

**The cost of being slightly wrong goes up.** In the opening a half-point
misjudgement is absorbed by fifty moves of play. On move fifty-six it is the
result. This is the shape of the complaint that started the feature: a player
reaching 80–90% win probability and then losing to a single move near the end.
The win-probability bar was not lying — it was reporting an estimate at exactly
the point where an estimate stops being good enough.

There is measured evidence for the same thing on the agent's own side.
`docs/experiments.md` records the search-budget sweep: against Edax level 9 the
agent scores 40%, 38% and 38% at 800, 1600 and 3200 simulations — flat within the
intervals, while every weaker Edax level improves by about one level per doubling.
Past 800 simulations something other than search depth is deciding those games.
Endgame precision is the standing hypothesis.

---

## The solver

`src/reversi/endgame/solver.py`. Pure Python, no torch, so it runs anywhere the
tests do.

It returns the **final disc difference under perfect play** — not "who wins".
That distinction is the whole reason it exists rather than the 4×4 solver already
in the test suite: "this move still wins, by 2" and "this move wins by 20" are
different lessons, and a solver that returns +1/0/−1 cannot tell them apart.

### The search

Negamax with alpha-beta pruning. *Negamax* is the usual way to write a two-player
search when the two players want opposite things: instead of one routine that
maximises and another that minimises, there is one routine, and each level negates
what the level below it returned. *Alpha-beta* is the pruning rule — the search
carries a window of scores still worth knowing about, and stops examining a branch
the moment it can prove the branch cannot change the answer. Neither changes the
result; both change how long it takes to get it.

Three implementation choices are the difference between seconds and minutes.

**It works on raw integers, not on `State`.** `rules.apply` re-validates the move
and builds an error string on the failure path, `rules.legal_actions` allocates a
list at every node, and `rules.is_terminal` runs move generation *twice*. None of
that is wrong — it is what you want at the edges of the system, where a bad move
should be caught loudly — and all of it is waste in a loop that runs millions of
times. Inside the search a position is two integers, and legality is guaranteed by
construction because every move came out of the generator.

**Terminal is detected with one extra generation, not two.** Correctness contract
C3 says a position is over exactly when neither side can place. So the search
generates its own moves; only if it has none does it ask whether the opponent has
any; and only if they have none is the game over. A pass is not stored as a
position of its own, because it is the same position seen from the other side.

**The transposition table never expires.** A *transposition table* remembers
positions the search has already evaluated, since the same position is often
reached by different move orders. A normal search can only reuse an entry at the
same remaining depth, because its stored value means "worth X if you search six
more plies". This search always runs to the end, so a position's value is a
property of the position alone. One table serves the whole solve, including across
different first moves — which is most of why ranking every move at the root costs
far less than solving one move nine times over.

Move ordering matters because alpha-beta prunes more when good moves come first.
Deep in the tree, moves are ordered by how few replies they leave the opponent —
the classic Othello heuristic. Near the leaves that ordering costs more than it
saves, so a static ordering by shape is used instead: corners first, because a
corner can never be flipped once taken; the squares beside a corner last, because
playing one hands the corner over. Both are tiebreaks, not evaluations — the
search still computes the true value of every move.

### `solve_root` deliberately does not prune between root moves

A puzzle needs the whole table, not the best entry. So each root move is searched
with a full window and comes back **exact**, rather than as a bound meaning only
"no better than the one before". That costs far less than it sounds, because the
transposition table is shared across root moves and never expires an entry.

### What it costs

Measured by `bench/endgame_bench.py`, on positions drawn from **real play** rather
than random boards. That distinction is not fussiness: a random position with
twelve empty squares has a sparse, scattered shape that searches far faster than
anything a game actually reaches, so benchmarking on random boards would produce
flattering numbers that predict nothing.

| empty squares | median | worst |
|---|---:|---:|
| 8 | 0.07 s | 0.13 s |
| 10 | 0.64 s | 1.42 s |
| 11 | 1.43 s | 3.34 s |
| 12 | 4.46 s | 8.34 s |
| 13 | 10.6 s | 14.7 s |
| 14 | 24.3 s | 65.9 s |

About ×2.4 per extra square. `MAX_EMPTIES` is set to 16 from this curve. It is a
**guard rail, not a capability claim**: the algorithm would happily start on a
position with thirty empty squares and never come back, so the solver refuses
above the limit rather than appearing to hang.

---

## How it is known to be right

A solver that is a little bit wrong is worse than no solver. It will tell a player
their winning move loses, under an interface that says the answer is certain, and
they will believe it and learn the wrong thing. So it is checked four ways, and
the fourth found something real.

| # | Anchor | Result |
|---|---|---|
| 1 | Every reachable 4×4 position, against the exhaustive solver | agrees everywhere |
| 2 | Terminal positions, against `scoring.score_margin` | exact |
| 3 | The negamax identity, including across a forced pass | holds |
| 4 | **Edax 4.6, on its own problem sets** | **exact on every position** |

Anchor 1 earns its cost because the 4×4 solver in `tests/unit/test_solved_4x4.py`
shares no pruning, no ordering and no caching with this one — so it shares none of
the ways this one could be wrong. Anchor 3 is where contract C3 either holds or
quietly does not. It also reproduces the known 4×4 result with detail the old
solver could not express: white wins **by 8 discs**, and all four of black's
opening moves lose by exactly 8.

### And a third implementation, in the browser

The page plays endings out with its own search (`web/.../engine/endgame.ts`),
which is a third implementation of the same idea and could be wrong in its own
ways. It is held to the standard the rules are: a generated `endgame` fixture
carries the exact value of a position and of every move in it, CI regenerates it
and compares byte for byte, and the sixty shipped puzzles are checked as well
because they go deeper than the fixture does. It agrees with the Python solver on
every move of every position.

That check matters more here than a duplicate-code complaint would suggest. The
browser search deliberately does **not** share the engine's board type: immutable
pairs are right at one position per move and wrong at a hundred thousand
positions per search, where allocation alone would dominate. The duplication is
the price of that, and the fixture is what stops the two drifting apart.

### The two scoring conventions

The first comparison against Edax had 36 of 40 positions agreeing. The four
failures had Edax's magnitude larger by exactly 1, 5, 1 and 1 — precisely the
number of squares left empty at the end of the game.

**Edax awards the leftover empty squares to the winner. This project awards them
to nobody**, which `scoring.result` states in as many words and which every
published number in this repository already uses.

The dangerous part is the shape of the failure. On the roughly 90% of positions
whose perfect play fills the board, the two conventions agree exactly — so a check
that ignored the difference would have **passed**, while hiding every case where
it mattered. They also optimise different objectives, so they can prefer different
moves, which means this is not a display detail that could be patched at the edge.

The solver therefore implements both. Validation runs in Edax's convention;
everything shipped uses this project's, which is also what a player sees, because
the final board simply has empty squares on it. A second test pins the
relationship — awarding empties to the winner can only widen a margin, never
narrow it — so a future change to either rule surfaces as a failure rather than as
drift.

---

## What makes a good puzzle

An exact answer is not yet a puzzle. Most positions near the end of a game are
either won by every legal move or lost by all of them, and neither teaches
anything: one has no wrong answer and the other has no right one.

> **A puzzle is a position where the exact answer and the tempting move
> disagree.**

Two things are needed to say that, and the project already had both:

* the **solver** says what is true — the exact result of every legal move;
* the trained network's **policy** says what looks tempting, because it is a model
  of what a plausible player wants to play.

A position qualifies when the side to move has a win and an attractive move throws
it away. That is also a description of the complaint the feature answers.

"Attractive" has two forms, and they are two different players:

* **the network's first choice**, which is the mistake a reasonable player makes;
* **taking the most discs**, which is the mistake a beginner makes, and the one an
  endgame punishes hardest — discs flipped now can be flipped straight back.

For the second, *every* move that flips the maximum number of discs must lose.
`GreedyAgent` breaks ties at random, so "grabbing discs loses here" is only a fact
about the position when no tie-break saves it — and picking one move would make a
committed artifact depend on a random draw.

The reasoning behind this rule, and the alternatives rejected, are in
[ADR-0008](decisions/ADR-0008-what-counts-as-a-puzzle.md).

---

## The curriculum

Five stages, by how many squares are still empty. That axis is the honest one and
the legible one: with four empty squares a player can count to the end in their
head, and with thirteen they cannot. It is also visible on the board, so somebody
can *see* why stage 4 is harder than stage 1 rather than being told.

| Stage | Empties | What it teaches |
|---|---|---|
| 1 | 4–5 | Count to the end |
| 2 | 6–7 | Who moves last in a region |
| 3 | 8–9 | The same counting across two regions |
| 4 | 10–11 | Giving discs away to keep the moves that decide it |
| 5 | 12–13 | Everything at once |

**Stage**, not level. The app already calls the six opponents Level 1 to Level 6,
and a control panel saying "Level" about two unrelated things is a bug waiting to
be filed as confusion — this repository was bitten by exactly that when "Strong"
was both a ladder rung and a thinking time.

Inside a stage the order comes from a score **derived** from the solve and the
network, never assigned:

1. **How narrow the win is.** One winning move out of eight is a far harder ask
   than five out of eight, and it is exact rather than estimated. The dominant
   term, because it is the only one that measures the position itself.
2. **How tempting the losing moves are** — the share of the network's probability
   resting on moves that throw the win away. A position whose natural move happens
   to be the right one is easy however few winning moves it has.
3. **Whether grabbing discs loses** — a flat bonus, weighted least despite
   teaching the most useful lesson, because it says something about one particular
   beginner habit rather than about how hard the position is to read.

The result is an integer on 0–100, so a shipped ordering cannot drift on a float
comparison between two runs or two languages. A test asserts the ordering is
monotonic within each stage and that stage bands do not overlap; that is the test
that keeps it a progression rather than a pile that happens to be sorted today.

---

## Mining

`reversi puzzles` plays games with the trained network at several difficulty
levels, harvests positions in the stage bands, and keeps the ones that qualify.

**It screens before it solves.** Ranking every move is what a puzzle needs and
roughly five times the work of deciding whether a position is worth keeping. So
each candidate is screened with one or two plain solves — is this position even
won, and does the tempting move lose it — and only a survivor pays for the full
root solve. The cheapest question that rejects the most candidates goes first:
most positions are simply not won by the side to move.

**Quotas are per stage.** A global top-N would quietly return a file of nothing
but the hardest positions found, which is the opposite of a progression — and the
hardest are also the deepest, so it would cost the most to produce.

**One game may contribute two puzzles.** Positions from the same game share most
of their board; without a cap, a handful of games would supply the whole
curriculum and a player would meet the same shape five times running while the
file still looked varied from outside.

The run that produced the shipped file: **53 games, 298 positions screened, 60
puzzles, 12 per stage**, about 30 minutes on a laptop CPU. 55 of the 60 are
positions where grabbing the most discs loses; 13 are positions where the
network's own first choice loses. That split is itself a measurement — the shipped
network is good enough at an ending that its top move rarely throws the game away,
so the disc-grabbing trap carries most of the selection.

### Why nothing diffs the generated file

The five engine fixtures are regenerated in CI and byte-compared, which is what
makes "generated, never typed" enforced rather than merely intended. That check
cannot work for puzzles and would be worse than useless if it looked like it did:
mining plays real games with the network, so which positions turn up depends on
torch floating-point arithmetic that is not promised to be identical across
machines or library versions. A diff against a re-run would fail for reasons that
are not mistakes, and the honest response to a check that fails for non-reasons is
to delete it.

What guards the file instead is stronger. **Every shipped puzzle is solved again
from its board**, its stored table compared move by move, and its stored line of
play replayed to confirm it reaches the promised score. Small puzzles run in the
fast test lane; the whole file runs in the slow lane. A hand-edited puzzle file is
dangerous because it would teach a wrong move — re-solving catches exactly that,
where a diff would only notice the file had been touched.

`docs/puzzles/endgame.obf` carries the same positions with every move's score in
**Edax's** convention, so `wEdax -solve` on it checks the answers against an
outside engine rather than merely re-solving them:

```bash
cd tools/edax && ./wEdax-x86-64.exe -solve ../../docs/puzzles/endgame.obf
```

All 60 agree.

---

## The page

`/puzzles/`. One question, asked sixty times: *you are winning — win it.*

You are handed an ending you are ahead in and you **play it out to the last
square** against an opponent that cannot be improved on. Solved means won.

The first version of this page graded one move and stopped, and it was the wrong
shape. Being told "+6" asks a player to trust a number; watching the position
resolve into a win they can count on the board shows them one. Worse, a single
wrong move ended the exercise with a verdict and no lesson — which is the exact
complaint the whole feature exists to answer.

**The page says nothing about the position while the ending is in progress.** The
exact value of every move is milliseconds away, and putting it on screen would
make the page playable by watching a number instead of by calculating. The whole
account arrives at the end, in a dialog that names the move that threw the win
away — which is the only place it can be said, and what makes the silence
affordable.

**That needs an exact search in the browser**, which is a third implementation of
the same idea and the one genuinely uncertain piece of engineering here. It was
measured before it was written: the single search a reply needs costs about forty
milliseconds at thirteen empty squares, against roughly two seconds for the same
question in Python, because tight integer bit-twiddling is what JavaScript does
far better than CPython. Every later move is cheaper, since the board only fills
up. It runs in a worker, so a burst of integer work cannot freeze a tab.

Precomputing the continuations instead was never viable. The opponent's replies
are forced, but the player's are not, so a stored tree branches on every move
they make — roughly fifteen thousand positions per puzzle at the deepest stage,
times sixty.

**Afterwards**, the stored line of perfect play can be stepped out to the end, and
every opening move's exact result is one click away. Both only after the ending is
over: shown during it, the ranking would be the answer key.

**Nothing is locked.** A gate would be the obvious reading of a staged
progression and the wrong call for a page strangers land on: hiding stage 4 behind
twelve solved puzzles mostly stops a visitor from seeing the interesting ones. The
order is carried by arrangement and labelling instead of by refusing input.

Progress is remembered per browser, keyed by the position itself, so a regenerated
curriculum keeps what survives it and drops what does not. It is a bookmark, not a
claim: the roadmap cut records and streaks because a tally kept in a visitor's own
browser can never be verified, and that objection does not apply to remembering
which puzzles somebody has already seen.

---

## What this does not do

**The agent does not use the solver when it plays.** Nothing about how any
difficulty level chooses a move has changed, which is why every published rating,
the difficulty calibration and the `models-v2` release all stand untouched. The
solver is analysis.

That is a deliberate deferral, not an oversight. Wiring it into play would very
likely lift the level-9 plateau — and it would invalidate the ladder, the
calibration and the Edax table, all of which would have to be re-measured. It is a
separate decision with a cost attached, and the level-9 hypothesis is now testable
without having been tested.

**It does not solve arbitrary positions.** Sixteen empty squares is the limit, set
from the measured curve rather than from the algorithm.

**It does not mark up a game you played elsewhere.** The page solves the ending
you are in, not one from your own history against the agent. That wants stored
games and a way to get them here, and is the most personal version of this
feature rather than a part of it.

---

## Where things are

| | |
|---|---|
| the solver | `src/reversi/endgame/solver.py` |
| mining and the curriculum | `src/reversi/endgame/puzzles.py` |
| the cost curve | `bench/endgame_bench.py` |
| the command | `reversi puzzles` |
| the shipped file | `web/src/games/reversi/engine/__fixtures__/puzzles.json` |
| the Edax cross-check | `docs/puzzles/endgame.obf` |
| the browser's search | `web/src/games/reversi/engine/endgame.ts` |
| its generated expectations | `web/src/games/reversi/engine/__fixtures__/endgame.json` |
| the page | `web/src/games/reversi/puzzles/` |
| tests | `tests/unit/test_endgame_solver.py`, `tests/unit/test_endgame_puzzles.py`, `web/tests/endgame.test.ts`, `web/tests/puzzles.test.ts` |
