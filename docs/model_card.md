# Model card: reversi-zero 1.0, generation 120

What this network is, how it was made, how strong it is, and what it cannot do.

**In one line:** a 458,761-parameter policy-value network that learned 8×8
Othello from nothing but games against itself, in 8 hours 20 minutes on one L40S.

This replaces the network shipped at 0.0, and is **+130 Elo** stronger, measured
in a single round robin containing both rather than by comparing two tournaments.

---

## What it is

| | |
|---|---|
| Task | 8×8 Othello (Reversi), full-strength play |
| Architecture | pre-activation ResNet, 6 blocks × 64 channels, three heads |
| Parameters | 458,761 |
| Input | 3 planes of 8×8: my discs, opponent's discs, my legal moves |
| Outputs | 65 policy logits (64 squares + PASS), one value in [−1, +1], 64 ownership predictions |
| Size on disk | 1.9 MB as PyTorch, 1.8 MB as ONNX |
| Intended use | playing Othello, as a demonstration of self-play learning |

**The third head is 65 parameters**, and it is the largest single gain the
project measured — worth about as much as a 6.7× bigger network, at no extra
self-play cost. It predicts which side will own each of the 64 squares when the
game ends, which gives the shared trunk 64 training signals per position instead
of one. It is an auxiliary target: nothing at play time reads it, and the ONNX
export does not even emit it. What it bought was a better trunk, not a better
value estimate — the diagnostic in `docs/experiments.md` records that ownership
turns out to be nearly unpredictable in Othello until the endgame, which is the
opposite of why it was tried.

The network never sees colour. Both planes and the value are from the point of
view of whoever is about to move, so black-to-move and the colour-swapped
white-to-move position are *the same input* — the network learns Othello once
rather than twice. See ADR-0002.

The network alone is not the player. It supplies a fast opinion; a PUCT tree
search improves on that opinion before a move is chosen, and how much search is
run is what the difficulty levels vary.

---

## Training data

**Self-play only.** No human games, no opening book, no hand-written evaluation
function, no positions from any external source. The agent began from randomly
initialised weights and played itself.

| | |
|---|---|
| Games | 72,000 |
| Positions in the final replay window | 1,042,245 across 660 shards |
| Generations | 120 |
| Search per move during self-play | 200 simulations |
| Exploration | Dirichlet noise at the root, ε = 0.25, α = 1.0 |
| Opening randomisation | sample moves for the first 15 plies |

Positions where only one move is legal are not stored: there is nothing to learn
from a forced move, and keeping them would bias the training set toward
positions that teach nothing.

**Provenance is recorded per run.** Every run directory holds the resolved
config with its hash, the git commit and diff, the environment, the seed and the
command line. No run can exist without them; `scripts/validate_run.py` fails if
any field is missing.

---

## Training compute

| | |
|---|---|
| Hardware | one NVIDIA L40S (48 GB) on the CSSE Slurm cluster, 24 CPU cores |
| Wall clock | 8 hours 20 minutes, across chained 24-hour jobs |
| Software | Python 3.11.2, PyTorch 2.6.0+cu124, CUDA 12.4 |
| Optimiser | SGD, momentum 0.9, weight decay 1e-4 |
| Learning rate | 0.02, 200 warmup steps, cosine decay to lr/20 |
| Batch size | 1024 |
| Gradient steps | 96,000 (800 per generation) |
| Auxiliary loss | ownership, weight 1.0, masked to played positions |
| Replay window | sliding, with a per-generation cap |
| Augmentation | a random one of the 8 board symmetries per sample |

This is a small amount of compute by the standards of the method. It is one
GPU for an evening, and the results below should be read in that light.

The run stops and resumes cleanly between generations: a job near its wall clock
is signalled, finishes the generation it is on, checkpoints, and the next job in
the chain picks it up. That is what makes 120 generations reachable under a
24-hour limit, and it was verified end to end before the run rather than assumed.

---

## Measured strength

Every number here comes from colour-balanced games with a seeded opening book and
no exploration noise. Ratings are Bradley–Terry, fit over the whole result matrix
at once, with 95% bootstrap intervals, anchored so random play = 0.

### Against the baselines

Round robin, 13 entrants, 78 pairings of 100 games — 7,800 games in one fit
(`docs/ratings/release-one-scale.json`):

| | Elo | 95% interval |
|---|---:|---|
| **Generation 120** | **1004** | 949 – 1070 |
| Generation 100 | 1002 | 945 – 1069 |
| Generation 114 | 987 | 932 – 1060 |
| Generation 65 | 928 | 868 – 992 |
| *0.0's generation 60* | *874* | *819 – 938* |
| Generation 35 | 862 | 807 – 928 |
| Generation 20 | 765 | 710 – 828 |
| Generation 15 | 722 | 670 – 785 |
| Generation 5 | 516 | 469 – 574 |
| Minimax depth-4 | 482 | 432 – 544 |
| Greedy | 396 | 336 – 461 |
| Minimax depth-2 | 317 | 265 – 375 |
| Random | 0 | — |

Generation 120 takes 95% off the depth-4 alpha–beta search and 80% off greedy.

**The learning claim:** generation 120's interval lies entirely above generation
35's, which lies entirely above generation 15's, which lies above generation 5's.
Three strict steps, not point estimates in a pleasing order — and not the loss
curve, which is never cited as strength evidence anywhere in this repository.

**The replacement claim:** +130 Elo over the network 0.0 shipped, which is in the
table above because it played in this tournament. A 480-game match on that
pairing gives 70.3% [66.1%, 74.2%], and the advantage holds with either colour.

**A rating in this table is not a property of a network.** Adding two generations
to this very round robin moved every strong entrant by about 40 points without a
single one of their games being replayed. What survives a change of field is the
*difference* between entrants that both played the same opponents. So these
numbers may be compared with each other and with nothing else — including the
0.0 model card's, whose generation 60 reads 877 there and 874 here by
coincidence.

### The difficulty ladder

Same checkpoint at four search budgets, 21 pairings × 300 games
(`docs/ratings/difficulty-calibration-gen120.json`):

| Level | Simulations | Elo | 95% interval |
|---|---:|---:|---|
| Casual | 16 | 544 | 508 – 585 |
| Club | 64 | 821 | 774 – 874 |
| Strong | 256 | 1186 | 1120 – 1254 |
| Max | 800 | 1381 | 1302 – 1457 |

Adjacent rungs differ by 277, 365 and 195 Elo with no overlapping intervals. All
from one network, so the separation is a property of the search rather than of
four different models.

Every gap is wider than it was on the 0.0 network, where the same four settings
gave 192, 267 and 163. Nothing about the settings changed; search improves on a
prior, so a better prior means each doubling of search buys more.

This is its own fit and its own scale. Greedy rates 396 in the round robin above
and 199 here — one frozen opponent, two tournaments, two coordinates.

### Against an outside engine

The one measurement that does not depend on a baseline written for this project.
At 256 simulations, against [Edax 4.6](https://github.com/abulmo/edax-reversi),
80 games per level:

| Edax level | score | 95% interval | |
|---:|---:|---|---|
| 5 | 71.2% | 60.5 – 80.0 | wins |
| 6 | 59.4% | 48.4 – 69.5 | even |
| 7 | 41.2% | 31.1 – 52.2 | even |
| 8 | 21.9% | 14.2 – 32.1 | loses |

So it sits between levels 6 and 7 — two levels better than the 0.0 network, which
was even at level 5 and lost to everything above it.

---

## Limitations

**It is not transitive.** Generation 120 takes 95% off the depth-4 search but
only 80% off greedy, which the same tournament rates 86 Elo *weaker* than depth-4.
A narrow self-play training distribution is the likely cause: the agent has seen
very few of the positions a disc-maximising player steers into. Smaller than it
was at 0.0, where the two figures were 100% and 63%, but **not fixed** — and it
is recorded in `docs/experiments.md` rather than smoothed over.

**It plateaued.** Generations 100, 114 and 120 are indistinguishable: every
head-to-head between them spans 50% (48.0%, 49.0%, 50.5%). Generation 120 ships
because it is the run's last checkpoint, not because it is its best one — the
measurement cannot tell which of the three is best, and picking the highest point
estimate would be selecting on noise.

**This checkpoint is the best of three seeds.** Two replicates of the same recipe
were trained (runs 7 and 8 in `docs/experiments.md`), and this one is the
strongest: it beats run 7's generation 120 in a direct 1000-game match, 56.8%.
That matters for how the numbers are read. A claim about *the recipe* — "the
ownership head is worth +74 Elo" — is a best-of-three figure and is said to be
one. A claim about *this network* — "+130 over what 0.0 shipped" — is a direct
measurement of the weights being shipped and carries no such selection.

**The easy levels are inconsistent, and it is now measured.** Casual and Club
sample among their top few moves rather than playing the best one, which is what
makes them weak rather than stupid. A player reported them as erratic, and the
measurement located the effect: it is *within* games, not between them. Block
scores vary no more than chance, but Casual declines its own best move 27% of the
time, with a 99th-percentile drop of 0.295 on a scale where 1.0 separates winning
from losing — about one move in a hundred costing nearly a third of a win, while
the median decline costs nothing.

So the tail is real and it is now a number rather than an impression. **Not
changed**: how heavy a tail is too heavy is a judgement about the product, and
the lever if it is judged too heavy is the guardrail, not the search budget. See
`docs/ratings/difficulty-spread-gen120.json`.

**Only four of the twenty-four offered combinations were rated.** Calibration ran
on generation 120 alone. The web app lets the level and the thinking time be
chosen independently, so a pairing like generation 5 at Club carries two ratings
from two different round robins and no measurement of the combination itself.

**The ratings are relative.** The scale is anchored at a random player written
for this project and shaped by a minimax written for this project. It is not
comparable to published Othello Elo.

**8×8 only.** The engine is size-parametric and 4×4 and 6×6 work, but this
network is trained for 8×8 and its input and output shapes are fixed to it.

**The agent does not use an endgame solver when it plays.** Strong Othello
programs solve the last ~20 plies exactly. This one searches them like any other
position -- the network guesses, a tree search improves the guess, and it stops --
which is where a classical engine gains the most on it.

An exact solver now exists in the repository (`src/reversi/endgame/`, validated
against Edax) and powers the endgame puzzles, but nothing in the playing path
calls it. That is deliberate: wiring it in would very likely lift the level-9
plateau recorded in `docs/experiments.md`, and it would invalidate every rating on
this card, the difficulty calibration and the Edax table, all of which would have
to be measured again. See `docs/endgame.md`.

---

## What it should not be used for

* **As an opponent that adapts.** It plays the same way every game at the top two
  difficulty levels, which are deterministic by design.
* **As an Othello authority.** It is beaten comfortably by the reference engine
  at moderate settings. It is a demonstration of a method, not a strong program.
* **As a general Reversi engine.** Rule variants — different starting positions,
  different board sizes with a trained network, scoring changes — are outside
  what it was built or tested for.

---

## Reproducing this

```bash
uv sync --extra cu124                      # or --extra cpu
uv run reversi train -c configs/full8x8_e2_ownership.yaml --generations 120
```

On a cluster with a wall clock, chain the jobs instead — `slurm/submit_chain.sh`
— and the run will stop between generations and resume where it left off.

The run is seeded, and every seed derives from `(run seed, generation, worker,
game index)`, so a resumed run replays the games it would have played. **Bitwise
determinism across machines is an explicit non-goal** — cuDNN autotuning, atomic
accumulation order and process scheduling all make it unattainable, and pursuing
it would cost more than it returns. Evaluation *is* deterministic: single
process, no noise, temperature 0, fixed opening seeds.

Expect a different network from a different seed, and expect the strength
conclusions to hold rather than the weights to match.

---

## Provenance

| | |
|---|---|
| Run | `e2-ownership`, on the CSSE Slurm cluster (node `gus`) |
| Config | `configs/full8x8_e2_ownership.yaml`, sha256 `c2afa6a8f12f4354…` |
| Seed | 1337 |
| Trained at commit | `67411db`, clean tree |
| Published as | `models-v2`, file `reversi-8x8-gen120-e2-ownership.onnx` |
| Checksum | sha256 `fb39b11358f6a65e…` for the PyTorch export; the ONNX file carries its own, in the sidecar beside it |
| Engine | frozen at commit `1108566`; every number above was measured against it |
| Licence | MIT, as the repository |

The published filename carries the run because release assets share one flat
namespace and more than one training run has a generation 60. See ADR-0007.

The rules engine is frozen deliberately. Changing it would invalidate every
strength number measured against it, so it is treated as fixed and any change
would mean re-running the evaluations rather than assuming they still hold.
