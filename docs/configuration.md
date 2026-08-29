# Configuration reference

Every setting in `configs/base.yaml`, what it does, why it has the value it has,
and what goes wrong if you change it.

`base.yaml` holds the 8×8 development settings. The other profiles only list
what they *change*:

- `smoke4x4.yaml` — tiny 4×4 board, for checking the pipeline works. Never a result.
- `dev8x8.yaml` — same as base; exists so run records say "dev8x8" explicitly.
- `full8x8.yaml` — the overnight run that produces the real result.

Load order: `base.yaml` → profile → `--set` overrides on the command line.

---

## The one idea that explains most of these numbers

**Generating games is slow. Training on them is fast.**

For one development generation:

| | Amount |
|---|---|
| Games played | 800 |
| Moves per game | ~58 (a full 8×8 board is 60 placements) |
| Network calls per move | 100 (one per simulation) |
| **Network calls to generate the data** | **~4,600,000** |
| Network calls to train on it | 400 steps × 512 = ~200,000 |

Generating the data costs roughly **20× more compute than learning from it.**

That single fact drives a lot of choices: the network is kept small, 32 games
run at once so the GPU gets big batches, and six separate processes play games
in parallel. If you only remember one thing about this config, remember this.

---

## Top level

```yaml
name: dev8x8
seed: 1337
run_root: runs
```

| Setting | Meaning |
|---|---|
| `name` | Which profile this is. Appears in the run folder name, so results are self-labelling. |
| `seed` | The single random number everything else is derived from. Each worker and each game gets its own seed computed from this one, so a run can be repeated, and the six workers don't accidentally play identical games. |
| `run_root` | Where run folders go. On the cluster the `RZ_RUN_ROOT` environment variable overrides this to point at scratch storage. |

---

## `game`

```yaml
game:
  board_size: 8
```

8 is standard Othello. The code also supports 4 and 6.

4×4 is used only to check the pipeline: it is small enough to train in ten
minutes on a laptop, so if the AI *doesn't* learn on 4×4, we know the training
code is broken before wasting a night of GPU time on 8×8.

---

## `net` — the neural network

```yaml
net:
  n_blocks: 4
  channels: 48
  value_hidden: 64
```

The network looks at a board position and outputs two things:

1. **A policy** — a score for each of the 65 possible actions (64 squares + "pass"), meaning "how promising does each move look?"
2. **A value** — one number from −1 to +1, meaning "how likely am I to win from here?"

It's a small ResNet (residual network, the standard image-recognition
architecture — a board is essentially an 8×8 image with a few channels).

| Setting | Meaning | Effect |
|---|---|---|
| `n_blocks: 4` | Depth: 4 residual blocks stacked up | More blocks = the network can combine information from further across the board, but every block slows down every one of those 4.6M calls |
| `channels: 48` | Width: 48 feature maps per layer | More channels = more patterns recognisable per layer |
| `value_hidden: 64` | Size of the hidden layer in the win-probability head | Minor; 64 is plenty for one number |

Together: about **180,000 parameters**. `full8x8` uses 6 blocks × 64 channels,
about 400,000.

**Why so small?** Two reasons.

1. *The bottleneck is speed, not capacity.* Doubling the network size roughly
   halves how many games we can generate per hour. At this board size, more
   games of experience beats a bigger network almost every time.
2. *An 8×8 board is small.* With 4 blocks, information can travel about 9
   squares across the board between input and output — already enough to see the
   whole board. Adding depth past that buys much less than it costs.

> **Why not a bigger network?**
> Because self-play data generation, not network capacity, is the bottleneck —
> it's about 20× the compute of training. A bigger network directly reduces how
> many games we produce per GPU-hour. The plan is to benchmark first and only
> grow the network if the Elo curve flattens *while* the GPU is underused.

---

## `mcts` — how the AI thinks about a move

This section is the heart of the algorithm. MCTS = Monte Carlo Tree Search.

Before every move, the AI builds a small search tree. Each **simulation** does
four things: walk down the tree picking moves, reach a position it hasn't seen,
ask the network to evaluate it, then send that evaluation back up the tree so
every move on the path updates its running average. After all simulations, the
move that got **visited most** is the one it plays.

### `n_simulations: 100`

How many times it does that per move. **This is the main strength dial.**

- More simulations → stronger play, proportionally more time per move.
- `dev8x8` uses 100, `full8x8` uses 300.
- The web app uses this to build difficulty levels: 16 simulations for Easy, 800 for Max, *from the same trained network*.

### `c_puct: 1.5`

The exploration constant. During a simulation, the tree picks the move that
maximises:

```
Q(a)  +  c_puct × P(a) × √(total visits) / (1 + visits to a)
```

- `Q(a)` = average result seen so far after playing move `a` — **what we've learned**
- `P(a)` = the network's prior score for `a` — **what we believe before searching**
- The visit-count fraction shrinks as a move gets visited more, pushing attention elsewhere

So `c_puct` sets the balance between trusting the network and checking
alternatives.

- **Too low** → the search just confirms whatever the network already thinks, and never discovers when the network is wrong.
- **Too high** → visits spread thinly across many moves, the tree stays shallow, and move choices get worse.

1.5 is the standard starting point; AlphaZero-family work generally lands
between about 1.25 and 2.5.

### `fpu_reduction: 0.25`

FPU = "First Play Urgency". This handles a specific edge case: what is `Q(a)`
for a move with **zero** visits, where we have no data at all?

The obvious answer, `Q = 0`, is a trap. Zero means "a draw." In a position where
we're losing, a draw looks *fantastic*, so the search abandons what it knows and
sprays visits over untried moves — staying shallow and playing worse.

Instead, an unvisited move starts at *the parent's current value, minus a small
penalty*: slightly pessimistic relative to what we already know. The search then
goes deeper on promising lines instead of spreading out.

> **Why this detail gets its own setting**
> FPU is a good answer — it's a one-line detail that measurably changes search
> quality, and the naive `Q = 0` version still runs perfectly happily, just weaker.

### `dirichlet_alpha: 1.0` and `dirichlet_eps: 0.25`

Exploration noise, added **only at the root of the search, and only during
self-play training games.**

The problem it solves: the AI trains on its own games. If it always plays what it
currently thinks is best, every game looks the same, it never tries alternatives,
and it can get permanently stuck on a mediocre strategy. Nothing pushes it out.

So at the root we blend random noise into the network's move preferences:

```
adjusted_prior = 0.75 × network_prior  +  0.25 × random_noise
```

- `dirichlet_eps: 0.25` is that 25% mixing weight.
- `dirichlet_alpha: 1.0` controls the *shape* of the noise. Small alpha puts nearly all the noise on one randomly chosen move (spiky — "seriously consider this one odd move"). Large alpha spreads it evenly across all moves. The usual rule of thumb is `alpha ≈ 10 / average number of legal moves`. Reversi averages roughly 8–10 legal moves, giving alpha ≈ 1.0. (For comparison, AlphaZero used 0.3 for chess with ~35 moves and 0.03 for Go with ~250 — the same rule.)

**This must be 0 whenever we're measuring strength** — evaluation matches, the
difficulty calibration, and the web app. Otherwise we'd be measuring a
deliberately handicapped agent and every Elo number would be wrong. The
evaluation code *asserts* it is zero rather than trusting the config.

> **Why self-play needs noise**
> Without it, self-play collapses into repeating the same game and the AI stops
> exploring. With it, roughly a quarter of the root's move preference is randomised
> each move, so the training data keeps covering alternatives. And it's switched
> off for evaluation, because the point of evaluation is to measure real strength.

### `temp_moves: 12` and `temp_init: 1.0`

After the search finishes, we have visit counts for each move. How do we choose
which one to actually play?

- **Temperature 1.0** — pick randomly, with probability proportional to visit count. Varied, sometimes suboptimal.
- **Temperature → 0** — always pick the most-visited move. Deterministic, strongest.

The schedule: play the **first 12 moves** at temperature 1.0, then switch to
"always pick the best" for the rest of the game.

- Early randomness → games start differently → varied training data.
- Late determinism → the rest of the game is played at full strength, so the recorded win/loss actually reflects good play.

Why 12 out of ~58? Enough to diversify openings, not so much that the AI plays
badly through the middlegame and learns from garbage.

> **The subtle part:** temperature affects
> only *which move gets played*. The training target we store is **always** the
> raw visit distribution, unaffected by temperature. Mixing those two up is one of
> the most common bugs in AlphaZero reimplementations — and it doesn't crash, it
> just quietly trains on the wrong thing.

---

## `selfplay` — generating the games

```yaml
selfplay:
  games_per_generation: 800
  n_workers: 4
  games_in_flight: 32
  max_generations: 15
```

### `games_per_generation: 800`

How many games to play before stopping to train. One generation = play 800
games → train → save a checkpoint → repeat with the improved network.

- Smaller → trains on fresher data more often, but more start/stop overhead.
- Larger → each cycle takes longer, and if the job gets killed you lose more work.

In `full8x8` this is **2500**, chosen so one generation takes about 20 minutes.
That's deliberate: the university job limit is 8 hours and jobs can be killed
early, so sizing a generation at 20 minutes caps the worst-case loss at 20
minutes rather than hours.

### `n_workers: 4`

Four separate OS processes playing games at the same time, one per CPU core.
`full8x8` uses 6 — the cluster gives us at least 8 cores, and we leave two for
the main process and for feeding the GPU.

They are fully **independent**: each plays its share of games, writes its own
data file, and exits. They don't talk to each other. That's a deliberate design
choice — the usual alternative (workers sending positions to a shared GPU
server over queues) is faster in theory but is the single most likely place for
the whole system to deadlock and hang overnight.

### `games_in_flight: 32`

**This is the most important performance setting in the file.**

Each worker plays 32 games *simultaneously*, interleaved. It advances all 32
searches one step, collects the 32 positions that need evaluating, and sends
them to the GPU **as a single batch**.

Why it matters: a GPU evaluating one position at a time is almost entirely idle —
the overhead of launching the work dwarfs the work itself. Feeding it 32
positions at once costs barely more time than 1 and does 32× the work. This
typically buys a 10–20× speedup with no change to the algorithm.

> **What makes self-play fast enough**
> Three things: bitboards for the game rules, batching 32 concurrent games into
> single GPU calls, and 6 parallel worker processes. And critically — we benchmark
> first and let the measurements decide what to optimise next, rather than guessing.

### `max_generations: 15`

Stop after 15 cycles. `full8x8` sets this high and instead stops on the job's
time limit, resuming from the last checkpoint the following night.

---

## `replay` — the training data pool

```yaml
replay:
  window: 100000
  per_gen_cap_factor: 2.0
  retain_shards: 30
```

### `window: 100000`

Keep the most recent 100,000 board positions to train on; forget older ones.

Why forget anything? Because positions from 20 generations ago were played by a
much weaker version of the network. Training on them drags the network backwards.

- **Too small** → the network sees the same few recent games repeatedly, overfits them, and training oscillates.
- **Too large** → a big fraction of the data is stale and improvement slows.

At 800 games × ~58 positions ≈ 46,000 new positions per generation, a position
stays in the window for roughly two generations before being pushed out.
`full8x8` uses 400,000 with 2500 games per generation — the same ratio.

### `per_gen_cap_factor: 2.0`

No single generation may supply more than 2× its fair share of a training batch.
A guard against one unusually large generation dominating the network's learning.

### `retain_shards: 30`

Keep 30 generations of data files on disk, delete older ones. Each file is
roughly 50–150 MB, so this is disk housekeeping — it stops a week-long run from
silently filling the shared filesystem.

---

## `train` — the learning step

```yaml
train:
  steps_per_generation: 400
  batch_size: 512
  optimizer: adamw
  lr: 0.001
  momentum: 0.9
  weight_decay: 0.0001
  warmup_steps: 200
  lr_floor_divisor: 20.0
  grad_clip: 5.0
  value_loss_weight: 1.0
  symmetry_aug: true
  checkpoint_every_steps: 200
```

### `steps_per_generation: 400` and `batch_size: 512`

400 gradient updates per generation, 512 positions each = 204,800 samples drawn
from a 100,000-position window. So each position is used about **twice per
generation**, and roughly **four to five times total** before it ages out.

That reuse rate is the thing to tune. Too many steps and the network memorises
the current window instead of learning general patterns; too few and we throw
away data we paid GPU-hours to generate.

### `optimizer: adamw`, `lr: 0.001`, `momentum: 0.9`

AdamW for development because it converges quickly with little tuning.
`full8x8` switches to **SGD at learning rate 0.02** — slower to tune, but it
generalises better, and it's what the AlphaZero line of work uses.

`momentum` applies only to SGD; AdamW ignores it. It's listed here so the
setting exists in one place rather than appearing only in one profile.

### `weight_decay: 0.0001`

Standard regularisation — mildly penalises large weights so the network prefers
simpler patterns and overfits less.

### `warmup_steps: 200`

Start at a near-zero learning rate and ramp up over the first 200 steps. A
freshly initialised network is random; a full-size first update can knock it
into a bad state it never recovers from. Warmup is cheap insurance.

After warmup, the learning rate follows a cosine curve downward.

### `lr_floor_divisor: 20.0`

The cosine decay bottoms out at `lr / 20` rather than at zero, so the network
keeps learning slowly at the end of a run instead of freezing.

### `grad_clip: 5.0`

If the total size of a gradient update exceeds 5, scale it down to 5. One
strange batch can otherwise produce an enormous update that destroys hours of
training. This costs nothing and prevents a rare, catastrophic failure.

### `value_loss_weight: 1.0`

The network learns two things at once, so the loss combines two terms:

```
loss = policy_loss  +  1.0 × value_loss
```

- **policy loss** — did the network's move preferences match what the search concluded?
- **value loss** — did its win prediction match how the game actually ended?

1.0 weights them equally (AlphaZero's choice). Raise it if win predictions stay
poor while move predictions are fine; lower it if the value head starts
memorising specific game outcomes.

### `symmetry_aug: true`

**A free 8× increase in training data.**

Reversi's rules are unchanged if you rotate the board 90°, 180°, 270°, or mirror
it — 8 symmetries in total (the dihedral group of the square). A position and its
rotation are the same problem with the same right answer.

So each stored position can be shown to the network in any of 8 orientations. We
pick one at random *each time a position is sampled* rather than storing all 8
copies — that saves disk space and means the network sees the same position from
a different angle each epoch.

> **Getting more out of limited data**
> Board symmetry. It costs a permutation of 64 indices and multiplies effective
> data by 8. The one detail to get right is that the "pass" action must stay put
> while the 64 square-actions are permuted — it isn't a square, so no rotation
> moves it.

### `checkpoint_every_steps: 200`

Save mid-training too, not just at generation boundaries, so a crash costs at
most 200 steps.

---

## `arena` — measuring whether it actually got better

```yaml
arena:
  every_n_generations: 5
  games: 200
  opening_plies: 4
  min_legal_after_opening: 2
  baselines: [random, greedy, minimax4]
  bootstrap_resamples: 1000
```

**This section is what makes the project a real experiment rather than a demo.**
Training loss going down does *not* mean the AI is playing better — the loss is
measured against targets the AI generated itself, so it can fall while strength
stagnates. The only way to know is to play matches and count.

### `every_n_generations: 5`

Run an evaluation every 5 generations. More often gives a finer strength curve
but spends GPU time on measuring instead of learning.

### `games: 200`

Games per matchup. **Must be even**, because each opening is played twice with
the colours swapped, so both agents play Black exactly half the time.

Why 200? Statistical precision. Win rates measured over few games are mostly
noise:

| Games | Roughly how precise the win rate is (95%) |
|---|---|
| 50 | ±14% |
| 200 | ±7% |
| 400 | ±5% |

At 200 games, "62% win rate" really means "somewhere between 55% and 69%" — good
enough to see progress. `full8x8` uses 400 for the final numbers.

### `opening_plies: 4`

The first 4 moves of every evaluation game are chosen randomly, from a fixed
seed.

Why: at evaluation, our AI plays deterministically (no noise, always the
best move) and so does minimax. Without randomised openings, **all 200 games
would be the identical game replayed 200 times** — one data point pretending to
be two hundred, and the confidence interval would be a lie.

Random openings make the games genuinely independent. Fixing the seed keeps the
whole thing reproducible. Playing each opening twice with swapped colours cancels
out any first-player advantage.

> **What makes the evaluation fair**
> Equal colours, a seeded random opening book so games are independent, each
> opening played from both sides, exploration noise disabled and asserted to be
> off, and results reported with confidence intervals rather than as a bare
> percentage.

### `min_legal_after_opening: 2`

Throw away a random opening if it leaves a player with fewer than 2 legal moves.
Forced positions don't tell us anything about who plays better.

### `baselines: [random, greedy, minimax4]`

The three opponents we measure against:

| Baseline | What it does | What beating it proves |
|---|---|---|
| `random` | Picks any legal move uniformly | The absolute floor. Losing to this means something is broken. |
| `greedy` | Flips as many discs as possible this turn | That it learned real strategy. Grabbing discs early is famously *bad* in Othello — it hands your opponent mobility — so beating greedy shows it learned to think beyond one move. |
| `minimax4` | Alpha–beta search 4 moves deep with a hand-written evaluation (corners are valuable, squares next to corners are dangerous, mobility matters) | **The real bar.** This is a genuinely competent classical opponent encoding decades of human Othello knowledge. |

These get **frozen** once written. A baseline that changes mid-project makes
every earlier measurement incomparable.

### `bootstrap_resamples: 1000`

For the Elo ratings, we resample the recorded game results 1000 times to compute
error bars. That turns "the AI is +250 Elo over greedy" into "+250 Elo, 95%
confidence interval +215 to +285" — which is a claim you can actually defend.

---

## `obs` — monitoring

```yaml
obs:
  tensorboard: true
  resource_sample_seconds: 1.0
  diagnostic_positions: 512
  wandb:
    enabled: false
```

| Setting | What it does |
|---|---|
| `tensorboard: true` | Live graphs while training runs |
| `resource_sample_seconds: 1.0` | Record GPU usage, memory, and CPU once per second — this is how we find out whether the GPU is sitting idle waiting for Python |
| `diagnostic_positions: 512` | A **fixed** set of 512 positions the network never trains on, used to measure quality consistently. Fixed so the numbers are comparable across generations. |
| `wandb.enabled: false` | Weights & Biases is an optional hosted dashboard. Off by default so no core part of the project depends on an external service. |

The two things measured on those 512 diagnostic positions are worth knowing:

- **Policy entropy** — how spread out the network's move preferences are. If it collapses toward zero the network has become overconfident and stopped exploring; if it stays at maximum it hasn't learned anything.
- **Value calibration** — when the network says "I'm 70% likely to win," does it actually win about 70% of those positions? A network can predict winners correctly while being badly calibrated, and calibration is what the web app's win-probability bar depends on.

---

## The five things to know cold for a demo

1. **`n_simulations`** — how hard the AI thinks per move. The main strength/speed dial, and the same dial that produces the four difficulty levels from one trained network.
2. **`games_in_flight`** — 32 games run at once so the GPU gets full batches instead of one position at a time. The single biggest performance decision.
3. **`dirichlet_eps`** — exploration noise during training so self-play doesn't collapse into one repeated game. Switched off, and asserted off, whenever we measure strength.
4. **`replay.window`** — only train on recent games, because old games came from a weaker network.
5. **`arena.games` + `opening_plies`** — how we prove improvement: enough games for a real confidence interval, with randomised balanced openings so those games are actually independent.

## The question to be ready for

> **"How do you know it's getting better?"**

The answer that separates this project from a tutorial:

*Not from the training loss.* The loss is computed against targets the AI
generated itself, so it can fall while playing strength goes nowhere. Instead we
play hundreds of matches against fixed baselines with balanced colours and
randomised openings, compute a confidence interval on the win rate, and fit Elo
ratings across every checkpoint. We only claim a later version is stronger when
its rating interval sits entirely above the earlier one's.
