# Model card: reversi-zero, generation 60

What this network is, how it was made, how strong it is, and what it cannot do.

**In one line:** a 458,696-parameter policy-value network that learned 8×8
Othello from nothing but games against itself, in 9.1 hours on one laptop GPU.

---

## What it is

| | |
|---|---|
| Task | 8×8 Othello (Reversi), full-strength play |
| Architecture | pre-activation ResNet, 6 blocks × 64 channels, two heads |
| Parameters | 458,696 |
| Input | 3 planes of 8×8: my discs, opponent's discs, my legal moves |
| Outputs | 65 policy logits (64 squares + PASS), one value in [−1, +1] |
| Size on disk | 1.9 MB as PyTorch, 1.8 MB as ONNX |
| Intended use | playing Othello, as a demonstration of self-play learning |

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
| Games | 36,000 |
| Positions stored | 2,077,597 |
| Generations | 60 |
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
| Hardware | one NVIDIA RTX A1000 Laptop GPU (4 GB), Intel i7-12700H |
| Wall clock | 9.1 hours |
| Software | Python 3.12.3, PyTorch 2.6.0+cu124 |
| Optimiser | SGD, momentum 0.9, weight decay 1e-4 |
| Learning rate | 0.02, 200 warmup steps, cosine decay to lr/20 |
| Batch size | 1024 |
| Gradient steps | 48,000 (800 per generation) |
| Replay window | sliding, with a per-generation cap |
| Augmentation | a random one of the 8 board symmetries per sample |

This is a small amount of compute by the standards of the method. It is what one
laptop can do overnight, and the results below should be read in that light.

---

## Measured strength

Every number here comes from colour-balanced games with a seeded opening book and
no exploration noise. Ratings are Bradley–Terry, fit over the whole result matrix
at once, with 95% bootstrap intervals, anchored so random play = 0.

### Against the baselines

Round robin, 8 entrants, 210 games each (`docs/crossgen.json`):

| | Elo | 95% interval |
|---|---:|---|
| **Generation 60** | **877** | 774 – 1028 |
| Generation 40 | 855 | 747 – 1018 |
| Generation 20 | 758 | 659 – 898 |
| Generation 5 | 547 | 467 – 686 |
| Minimax depth-4 | 523 | 434 – 653 |
| Minimax depth-2 | 334 | 250 – 459 |
| Greedy | 313 | 220 – 468 |
| Random | 0 | — |

Generation 60 beat the depth-4 alpha–beta search **30 games to nil**.

**The learning claim:** generation 60's interval lies entirely above generation
20's. That is what licenses "it got stronger" — not the loss curve, which is
never cited as strength evidence anywhere in this repository.

### The difficulty ladder

Same checkpoint at four search budgets, 21 pairings × 300 games
(`docs/difficulty_calibration.json`):

| Level | Simulations | Elo | 95% interval |
|---|---:|---:|---|
| Casual | 16 | 431 | 400 – 465 |
| Club | 64 | 623 | 588 – 664 |
| Strong | 256 | 891 | 845 – 943 |
| Max | 800 | 1053 | 1001 – 1115 |

Adjacent rungs differ by 192, 267 and 163 Elo with no overlapping intervals. All
from one network, so the separation is a property of the search rather than of
four different models.

---

## Limitations

**It is not transitive.** Generation 60 beats the depth-4 search 100% of the time
but the greedy baseline only 63%, while depth-4 beats greedy 97%. A narrow
self-play training distribution is the likely cause: the agent has seen very few
of the positions a disc-maximising player steers into. This is recorded in
`docs/experiments.md` and **not fixed**.

**It plateaued.** Generation 40 and generation 60 are indistinguishable — 56.7%
[39.2%, 72.6%] head to head. The value loss stopped improving at generation 5
(0.644) and had not moved by generation 60 (0.661) while the policy loss
continued to fall. More training at these settings would likely produce
generation 60 again.

**One run, one seed.** Nothing here separates "the method works" from "this seed
worked". A second run at a different seed would.

**The ratings are relative.** The scale is anchored at a random player written
for this project and shaped by a minimax written for this project. It is not
comparable to published Othello Elo.

**8×8 only.** The engine is size-parametric and 4×4 and 6×6 work, but this
network is trained for 8×8 and its input and output shapes are fixed to it.

**No endgame solver.** Strong Othello programs solve the last ~20 plies exactly.
This one searches them like any other position, which is where a classical engine
gains the most on it.

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
uv run reversi train -c configs/full8x8.yaml --generations 60
```

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
| Run | `20260827-030939-full8x8-60cfdda-s1337` |
| Config | `configs/full8x8.yaml` |
| Seed | 1337 |
| Engine | frozen at commit `1108566`; every number above was measured against it |
| Licence | MIT, as the repository |

The rules engine is frozen deliberately. Changing it would invalidate every
strength number measured against it, so it is treated as fixed and any change
would mean re-running the evaluations rather than assuming they still hold.
