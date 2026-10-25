# Step 1 — Pilot Results

**Applies to:** Step 1 of the [experiment design proposal](experiment-design-proposal-2026-10.md), run exactly as [`eval-protocol-frozen.json`](eval-protocol-frozen.json) specifies
**Played:** October 25, 2026, on a laptop CPU, 47 minutes with six worker processes
**Report:** [`results/pilot-2026-10.json`](results/pilot-2026-10.json) (committed). The 3,000 recorded games (70 MB) are under `runs/research/pilot-2026-10-games/` and are not committed; the tables below are produced from them by [`scripts/analyze_pilot.py`](../scripts/analyze_pilot.py).
**Gate A: the pattern persists**, with far more precision than the historical result had, and the predefined wipeout count turns out to describe it almost completely. Step 2 proceeds.

## 1. What was run

Six entrants in a round robin: three networks (Run 1 generation 60; the `control` run's generation 120, which is the same recipe trained longer; the `e2-ownership` run's generation 120, which adds an ownership head), Greedy, Minimax-d4 and Random. 15 pairings, 200 games each: 100 seeded 4-ply random openings, every opening played once with each colour. 50 search simulations per move, no exploration noise, seed 20261025. Every move was recorded, with the network's value estimate before and after search at each of its moves.

One line of code changed between freezing the protocol and running it: the tournament now caps numpy's BLAS library at one thread per worker process, because six workers each reserving memory for a 24-thread pool ran the laptop out of memory before the first game. Moves and search values were checked to be identical with and without the cap (commit `c26093d`). Nothing in the protocol changed.

## 2. The headline table

Scores are from the network's side: wins plus half the draws, over 200 games. Intervals are 95% Wilson intervals, as the report computes them.

| checkpoint | rating, Random = 0 [95%] | vs Greedy [95%] | vs Minimax-d4 [95%] | wiped out by Greedy, of 200 |
|---|---|---:|---:|---:|
| gen60 (Run 1, generation 60) | +527 [+489, +569] | **64.5%** [57.7%, 70.8%] | 95.8% [92.0%, 97.8%] | **70** |
| control120 (same recipe, generation 120) | +584 [+545, +627] | **64.0%** [57.1%, 70.3%] | 96.5% [93.0%, 98.3%] | **72** |
| gen120 (ownership head, generation 120) | +685 [+644, +732] | **82.0%** [76.1%, 86.7%] | 99.0% [96.4%, 99.7%] | **35** |
| Minimax-d4 | +335 [+304, +370] | 98.2% [95.3%, 99.4%] | | 2 |
| Greedy | +244 [+202, +291] | | | |

Random scored 1.0% to 1.5% against every network and against Minimax-d4, and 37.8% against Greedy. Nothing is broken.

**The historical result reproduces.** Run 1's generation 60 scored 63% against Greedy on 30 games in August; it scores 64.5% on 200 games here, under the same budget and opening design, and the interval now excludes everything below 57%. Against Minimax-d4 it scores 96%, so the ordering Greedy < Minimax-d4 < gen60 by rating holds while the margins do not line up: the network beats the stronger fixed opponent far more comfortably than it beats the weaker one. This is the pattern the proposal set out to check, and it is not noise.

## 3. Every loss to Greedy is a wipeout

A *wipeout* is a game that ends with one side holding no discs. Lengths below are moves after the 4-ply opening, passes not counted; a game that fills the board is 56 moves long.

| network | opponent | losses | of which wipeouts | moves at the wipeout: shortest / median / longest | losses that lasted 50+ moves |
|---|---|---:|---:|---|---:|
| gen60 | Greedy | 71 | **70** | 6 / 13 / 28 | **0** |
| control120 | Greedy | 72 | **72** | 6 / 15 / 36 | **0** |
| gen120 | Greedy | 36 | **35** | 5 / 13 / 24 | **0** |
| gen60 | Minimax-d4 | 8 | 1 | 48 (that one game) | 7 |
| control120 | Minimax-d4 | 7 | 0 | | 7 |
| gen120 | Minimax-d4 | 2 | 0 | | 2 |

Of the 179 games the three networks lost to Greedy, 177 ended with the network holding no discs, and not one lasted fifty moves. Half of the wipeouts came within 13 to 15 moves of the opening; the earliest came at move 5, close to the shortest game Othello allows. The networks' losses to Minimax-d4 look nothing like this: 16 of 17 went the distance and were lost on the final count.

**It is not that Greedy wipes everyone out.** Greedy wiped out Random 9 times in 200 games and Minimax-d4 twice. It wiped out gen60 70 times. Nor is it that the networks wipe opponents out as a matter of course: gen60 wiped out Random twice and Greedy seven times. Wipeouts are rare in this tournament everywhere except when a network plays Greedy.

So the "64%" has a shape. Against Greedy the networks win almost every game that reaches the middlegame and lose a third of their games before it starts. The historical description, "weaker against Greedy than against Minimax", was true and was the wrong description.

## 4. Colour

No pairing shows a colour effect the data can support. The colour difference below is the first-named agent's score with black minus its score with white, with a 95% bootstrap interval that resamples opening pairs rather than games, because the two games of a pair share a position.

| pairing | first agent as black | as white | black minus white [95%] | wiped out as black / as white (network) |
|---|---:|---:|---|---|
| Greedy vs gen60 | 36.0% | 35.0% | +1.0 pts [-13, +14] | 35 / 35 |
| Greedy vs control120 | 41.0% | 31.0% | +10.0 pts [-2, +22] | 31 / 41 |
| Greedy vs gen120 | 18.0% | 18.0% | +0.0 pts [-11, +11] | 18 / 17 |
| Minimax-d4 vs gen60 | 5.0% | 3.5% | +1.5 pts [-4, +7] | 0 / 1 |
| gen60 vs gen120 | 26.0% | 35.0% | -9.0 pts [-22, +4.5] | |

Every interval includes zero. gen60's wipeouts split 35 as black and 35 as white. The withdrawn historical claim of a 90%/50% colour split for gen60 against Greedy stays withdrawn: on 200 games gen60 scores 64% with black and 65% with white. The one difference worth a second look, Greedy doing ten points better with black against control120, is within what 100 opening pairs produce by chance and is not pursued further unless it recurs.

## 5. The two one-variable comparisons

The three checkpoints were chosen so that each step changes one thing.

| comparison | changes | vs Greedy | wipeouts by Greedy | rating |
|---|---|---|---|---|
| gen60 → control120 | 60 more generations of the same recipe | 64.5% → 64.0% | 70 → 72 | +527 → +584 |
| control120 → gen120 | ownership head added, same generation | 64.0% → 82.0% | 72 → 35 | +584 → +685 |

**More self-play does not touch it.** Sixty more generations bought the control about 57 rating points, and head to head it beats gen60 65% of the time, yet against Greedy it scores the same and is wiped out just as often. Whatever the network learns between generations 60 and 120, it is not how to survive an opponent that flips as many discs as it can.

**The ownership head halves it.** At the same generation, the network trained with the ownership head is wiped out 35 times instead of 72. The ownership head trains the network to predict which colour will own each square at the end of the game, which is sixty-four targets per position about *where* discs will end up rather than one target about *who* wins. A plausible reading is that this teaches the network to care about its disc count in a way the win/loss target alone did not; that is a hypothesis for Step 2, not a finding here. Both comparisons rest on one training run per recipe. The second seed of the ownership recipe (`e2-ownership-seed2`, Run 7) exists and is the cheap way to check that the second comparison is the head and not the seed.

## 6. What this changes for Step 2

Step 2's job was to find a repeatable, descriptively meaningful failure pattern. The pilot has handed it one before any analysis of positions: *the network loses to Greedy by being wiped out early, and only to Greedy*. The questions for Step 2 are now specific:

1. **What does the network believe while it is being wiped out?** Every recorded network move carries the value estimate before and after search. For each of the 177 wipeouts, plot that value over the last several moves. If it stays positive until the end, the network cannot see the danger of a small disc count at all; if it drops a few moves before the end, the problem is that search at 50 simulations finds no escape in time.
2. **Has the network ever seen a position like these?** The self-play replay data under `runs/` can be scanned for positions where one side holds three discs or fewer. If such positions are nearly absent from training, the "never learned it" explanation has a number behind it.
3. **Does more search help?** Replay the wipeout openings at 256 and 800 simulations. If the wipeouts vanish, the weakness is in how far the search looks; if they stay, it is in the value estimate, and no search budget fixes it.
4. **Is it the head or the seed?** Add `e2-ownership-seed2` generation 120 to the Greedy pairing (200 games, a few minutes).

Each of these is cheap and none requires training. Items 1 and 2 are analysis of data that already exists.

## 7. What this does not show

- It does not show *why* the ownership head helps, or that it is the only recipe change that would.
- It does not show that self-play's training distribution caused the weakness. Item 2 above would be evidence toward that, not proof of it.
- It does not generalise beyond Greedy. Whether other disc-hungry opponents produce the same wipeouts is a Step 3 question, and the held-out roster in [held-out-opponents.md](held-out-opponents.md) should include at least one such opponent for exactly this reason.
- It does not establish a non-transitive cycle. Every network beats Greedy; the finding is about the margin and the manner, not the direction.
