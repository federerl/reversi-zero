# Step 0 — Protocol and Data Audit

**Applies to:** Step 0 of the [experiment design proposal](experiment-design-proposal-2026-10.md)
**Date:** October 25, 2026
**Outcome:** the evaluation protocol for Step 1 is frozen in [`eval-protocol-frozen.json`](eval-protocol-frozen.json). One historical claim is withdrawn. Match recording was added to the arena so that Step 2 can look at games, not just scores.

## 1. What the historical Greedy result rests on

The proposal's motivating numbers come from Run 1's cross-generation tournament. The audit traced each quoted number to the report that produced it.

| Item | Found |
|---|---|
| Report file | `docs/crossgen.json`, committed in `089e333` (2026-08-28). A byte-for-byte different but numerically identical copy sits in `runs/20260827-030939-full8x8-60cfdda-s1337/arena/crossgen.json` (the run directory is not in git). |
| Protocol | 28 pairings, 30 games each, 4-ply seeded random openings, each opening played once per colour, seed 20260828. |
| Search budget | **50 simulations per move** for every network entrant, recorded in the report's `agent_specs`. Training ran at 200. The 1000-game matches in Run 3 also used 50. |
| Checkpoint | `gen_00060.pt` of Run 1. Its SHA-256 (`7fb9014e…`) matches `source_sha256` in `models/reversi-8x8-gen60.json`, so the published `models/reversi-8x8-gen60.pt` is a faithful export of the checkpoint that was rated. |
| Per-game records | **None.** The arena recorded only win/loss/draw counts, the colour split, and mean game length. Which moves were played, and what the search thought, were never written down. |

[`scripts/audit_run1_claims.py`](../scripts/audit_run1_claims.py) reads the report and prints each number quoted in `docs/experiments.md` next to the number the report holds. Its output on October 25:

| Quoted claim | Report | Status |
|---|---|---|
| gen05/20/40/60 vs Greedy: 90 / 70 / 70 / 63% | 90.0 / 70.0 / 70.0 / 63.3% | supported |
| gen05/20/40/60 vs Minimax-d4: 57 / 87 / 97 / 100% | 56.7 / 86.7 / 96.7 / 100% | supported |
| Minimax-d4 vs Greedy 97% | 96.7% (29 of 30) | supported |
| gen60 vs Greedy 95% interval [46%, 78%] | [45.5%, 78.1%] | supported |
| **gen60 vs Greedy: 90% as black, 50% as white** | **60.0% as black (9 of 15), 66.7% as white (10 of 15)** | **not supported** |

The colour-split claim is not in the report that the surrounding paragraph cites, and no other report of a gen60–Greedy pairing exists in the repository or in the run directory. The claim was written in the same commit that added the report, so it was unsupported from the day it was written. The only place the figures "90% as black and 40% as white" appear in the repository is as an illustrative example in the docstring of `src/reversi/arena/match.py`, which predates the tournament; the most likely explanation is that the example was mistaken for a result.

**Decision:** the claim is withdrawn in `docs/experiments.md`, with the correction kept next to the original text. The proposal's hypothesis H2 no longer cites a colour asymmetry as existing evidence; whether one exists is a question for Step 1.

## 2. Checkpoint identities

| Label | File | Source run | Source checkpoint SHA-256 | Architecture | Git commit |
|---|---|---|---|---|---|
| gen60 | `models/reversi-8x8-gen60.pt` | `20260827-030939-full8x8-60cfdda-s1337` (Run 1, laptop) | `7fb9014e1a6c…` (verified against the run directory) | 6 blocks × 64 channels, policy + value | `cf51284` |
| gen120 | `models/reversi-8x8-gen120.pt` | `e2-ownership` (Run 5, CSSE Slurm cluster) | `e32fbd58d0ac…` (verified against the run directory) | 6 × 64, policy + value + **ownership head** | `67411db` |
| control120 | `models/reversi-8x8-control-gen120.pt` | `control` (Run 3, CSSE Slurm cluster) | `72c39047b30d…` (exported from the run directory on October 25) | 6 × 64, policy + value | `543de27` |

The proposal called the second checkpoint "the 120-generation control checkpoint". It is not: `runs/control` and `runs/e2-ownership` are two different runs. Their configs differ only in the ownership head and its loss weight, so gen120 is "the control recipe plus an ownership head", and it is the network the 1.0 release ships. The proposal now names it correctly.

**Why a third checkpoint.** gen60 and gen120 differ in two ways at once: sixty more generations of training, and the ownership head. Any difference between them against Greedy could be either. `control` gen120 is Run 1's exact recipe (`full8x8.yaml`, no ownership head) trained to generation 120 on the cluster, so it sits between the other two and separates the two explanations:

| Comparison | Same | Differs | Answers |
|---|---|---|---|
| gen60 vs control120 | recipe | 60 more generations | does more self-play, on its own, change the result against Greedy? |
| control120 vs gen120 | generation | ownership head | does the ownership head change it? |

The published export was made with `reversi export` from `runs/control/checkpoints/gen_00120.pt`; the sidecar `models/reversi-8x8-control-gen120.json` records the source checkpoint's hash and the training commit. `models/` is not in git, so anyone reproducing the pilot re-exports from the run directory and checks the hashes in the frozen protocol.

Neither cluster run has a `git.json` in its run directory; their commits come from the export sidecars.

## 3. Determinism and cost, measured

Four games of each checkpoint against Greedy, played twice from the same seed with recording on, at 50 simulations on one CPU thread:

| Checkpoint | Same results | Same moves, every game | Same search values, every move | Time per game |
|---|---|---|---|---|
| gen60 | yes | yes | yes | 1.7 s |
| gen120 | yes | yes | yes | 3.3 s |
| control120 | yes | yes | yes | 3.1 s |

So the arena is reproducible from its seed down to the search's own numbers, as its design intends: no exploration noise, no temperature, one derived seed per game. Any difference between two runs of the frozen protocol is a difference in code, weights, or opponent, not in luck.

**Pilot cost estimate.** A six-entrant round robin (gen60, control120, gen120, Greedy, Minimax-d4, Random) has 15 pairings; at 200 games per pairing that is 3,000 games. The network pairings dominate at roughly 2–5 s per game, more when both sides search; Minimax-d4's own search adds to its pairings. On the laptop with six worker processes this is about an hour and a half. The CSSE cluster's CPU nodes would be faster; a GPU is not useful here, since the search asks about one position at a time.

## 4. A first look at how gen60 loses to Greedy

The determinism check's recorded games were examined, and then 20 games were played to look again (seed 20261025, 50 simulations). This is a **look, not a result**: 20 games, one seed, no interval worth quoting.

gen60 scored 15–5. Every one of the five losses was a **wipeout**: gen60 lost all of its discs, after 5, 8, 11, 18 and 21 plies beyond the opening. Every one of the fifteen wins went the full distance to a filled or nearly filled board. No loss looked like a close game lost at the end.

If this holds up in Step 1, it sharpens the question considerably. A wipeout is the one position type self-play almost never produces: two copies of the same network rarely let one side's disc count fall to zero, so the network may never have learned that a tiny disc count is fatal. Greedy, which flips as many discs as it can every move, is the opponent most likely to produce one. The historical "63%" would then be "wins almost every normal game, and gets wiped out in about a quarter of them". Step 2's failure analysis should start by counting wipeouts in the pilot's recorded games and plotting the search value in the moves before each one.

## 5. What was added to the code

The arena can now record games. `play_match` takes a `record` list and appends one `GameRecord` per game: the opening, every move, and for any agent that reports its search (the network agent now does), the network's value before searching, the searched value, and the visit distribution at each of its moves. `round_robin_parallel` takes a `games_dir` and writes one file per pairing; `reversi arena --record-games` turns it on. Recording does not change which moves are played, and a test checks that.

Verification: `ruff check`, `ruff format --check`, `pyright` (0 errors), `pytest -m "not slow and not gpu"` (657 passed).

## 6. The frozen protocol

[`eval-protocol-frozen.json`](eval-protocol-frozen.json) fixes, for Step 1:

- the three checkpoints by file and SHA-256;
- the opponents: Greedy, Minimax-d4 (its evaluation weights are fingerprinted in every report), Random as a sanity check; Edax 4.6 is a reference evaluator for Step 2, not a Step 1 opponent;
- 50 simulations per move, so results sit on the same scale as every historical number;
- 4-ply seeded openings, 100 opening pairs = 200 games per pairing, seed 20261025;
- recording on; the report is committed under `research/results/`, the games stay under `runs/` (they are a few megabytes per pairing and are not needed to check the report).

The command is in the file. Step 1 runs it and reads nothing else.

## 7. Open items

- Whether the cluster's CPU nodes are available for the pilot; the laptop suffices if not.
