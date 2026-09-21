# reversi-zero

An AlphaZero-style **Othello** system with self-play training, PUCT MCTS, calibrated difficulty
levels, and an interactive web app.

> **Othello or Reversi?** The same game. Reversi is the older name; Othello is the 1971 version
> that fixed the board at 8×8 and the opening at four discs crossed in the centre, which is what
> this implements. The game is called Othello anywhere a player or a reader meets it. `Reversi`
> stays as the identifier — the package, the module paths, the model filenames, the `/reversi/`
> route and the *Reversi Zero* name — because those are addresses rather than descriptions, and
> renaming them would orphan a published model release whose filenames carry their own
> provenance.

The agent starts from randomly initialised weights and learns **only** from self-play — no human
games, no opening books, no hand-written evaluation. The point of the project is not just that it
learns, but that the claim is *measured*: playing strength is established by tournaments with
confidence intervals and Bradley–Terry ratings, never by pointing at a training-loss curve.

> **Status: the 8×8 run is done, rated, and playable in a browser.**
>
> 120 generations, 72,000 self-play games, 8 hours 20 minutes on one L40S. In a 13-entrant round
> robin of 7,800 games, the shipped agent rates **+1004 Elo** (95% bootstrap interval 949–1070)
> against random play at 0. Its interval lies entirely above generation 35's, which lies entirely
> above generation 15's, which lies above generation 5's — the evidence the project set out to
> produce: it got stronger, and every step of the claim has error bars.
>
> It is also **+130 Elo over the network this project shipped at 0.0**, measured in that same fit
> rather than by subtracting two tournaments, and confirmed by a 480-game match it won 70.3%
> (95% interval 66.1–74.2%).
>
> **The agent runs in your browser.** The trained network is 1.8 MB, so instead of renting a
> machine to run it, the site hands a copy to each visitor. It is a static site: nothing to keep
> awake, nothing to overwhelm, no request that carries your game anywhere. See *Play it* below.
>
> **Both dials are measured.** Six levels, every adjacent pair separated: random at 0, a
> disc-counting rule at +396, then generations 5, 15, 35 and 120 at +516, +722, +862 and +1004.
> Four generations published out of eight rated, because those four are the ones whose intervals do
> not overlap their neighbours — a level you cannot tell from the one below is not a level.
>
> The four thinking times are measured separately, on the shipped network: Casual, Club, Strong and
> Max rate **+544, +821, +1186 and +1381** over 21 pairings of 300 games — gaps of +277, +365 and
> +195 against a requirement of 80, with no adjacent intervals overlapping. All from one checkpoint,
> so the separation is a property of the method rather than of four different networks
> (criterion S15).
>
> **It holds its own against a real engine.** Measured against
> [Edax 4.6](https://github.com/abulmo/edax-reversi), the reference Othello engine: the shipped
> network at 256 simulations **beats Edax level 5** 71.2% over 80 games (95% interval 60.5–80.0%)
> and is statistically even with **levels 6 and 7** — 59.4% and 41.2%, both intervals spanning 50%.
> It loses to level 8. That is the claim here that does not depend on a baseline written in this
> repository, and it is two levels better than 0.0 managed.
>
> **One thing is honestly not fixed.** The agent beats depth-4 minimax 100% of the time but greedy
> only 63% — a real non-transitivity that `docs/experiments.md` records and explains but does not
> resolve.

![The board mid-game against level 6. Legal moves show as dots, the panel names the level and how
hard it is, and reports the agent's own estimate of who is winning. A note beside the level opens to
give the checkpoint it plays, its rating and the 95% interval.](docs/figures/demo.gif)

*Playing level 6 in the browser. A player picks a numbered level; every level and every thinking
time still carries a measured rating and its interval, one click away rather than in the way. The
whole search runs locally, so nothing about the game leaves the page.*

## Quickstart

```bash
uv sync --extra cpu --extra dev --extra api --extra obs   # GPU cluster: --extra cu124
make test        # fast suite, CPU only, target < 90s
make quality     # ruff + pyright
```

On Windows, `.\make.ps1 test` is an equivalent shim for the same targets.

```bash
reversi config -c configs/full8x8.yaml     # resolved config + its sha256
reversi init-run -c configs/smoke4x4.yaml  # create a run dir and write provenance
python scripts/validate_run.py runs/*      # assert every run is reproducible

reversi train -c configs/smoke4x4.yaml     # 4x4 pipeline validation, ~8 min CPU
reversi train -c configs/full8x8.yaml      # 8x8 run (needs a GPU to be worth starting)
reversi arena --suite crossgen --run-id <id>  # rate a run's generations against Random / Greedy / Minimax
reversi calibrate models/reversi-8x8-gen60.pt  # measure the four difficulty levels
reversi serve                              # play against a checkpoint from a local server
```

## Play it

The site is a directory of static files. There is no server: the rules, the tree search and the
network all run in the visitor's browser.

```bash
cd web
npm ci
npm run dev            # http://localhost:4173 — rebuilds on save
npm test               # the engine against fixtures generated by the Python one
npx playwright test    # a whole game, in a real browser, against the built site
```

`npm run dev` builds and serves rather than running the Vite dev server: Vite's
dev-mode module transform breaks how `onnxruntime-web` loads itself, and the page
hangs at "Loading the agent…". `docs/web-app.md` has the details.

The networks are not in git. To play against the real agent locally, export them first:

```bash
uv run reversi export runs/<run-id>/checkpoints/gen_00120.pt models/reversi-8x8-gen120.pt
uv run reversi export-onnx models/reversi-8x8-gen120.pt \
    web/public/models/reversi-8x8-gen120-<run key>.onnx
```

Or fetch the published ones, which is what a deploy does and what the checksums are for:

```bash
cd web && npm run fetch-models
```

**What you can choose.** A level, 1 to 6, and for the levels a network plays, how long it thinks.

The ladder is not a difficulty dial with six positions. Each level *is* an opponent, and the order
they are in was measured rather than chosen: random play at level 1 (0 Elo), a disc-counting rule at
level 2 (+396), then four checkpoints from the self-play run — generation 5 (+516), 15 (+722), 35
(+862) and 120 (+1004). The ladder is computed from those ratings, so adding a checkpoint inserts a
level in the right place with no code change.

Eight generations were rated and four are published. The other four — 20, 65, 100 and 114 — have
intervals that overlap a neighbour, and a level you cannot tell apart from the one below makes the
whole ladder feel arbitrary. Rating more than you ship and letting the fit choose is the rule.

Levels 1 and 2 are not the network, and that is not a shortcut. Generation 5 already rates above the
depth-4 search it was measured against (+516 against +482), so there was no rung a new player could
beat; Random and
Greedy are the baselines the whole project was measured against, and they fill it in. The page starts
on level 2, which is beatable and needs no model downloaded at all.

Level and thinking time are separate controls on purpose: the level sets how good the agent's
intuition is, and the search budget sets how much it improves on that intuition before moving. The
difference between those two is the whole idea behind the method.

A level shows a number and a word. Which checkpoint it is, what it rates and how wide the interval
is sit under a note beside it — true, checkable, and not between a player and their move.

**How fast it is.** Measured in Chrome on a 20-core laptop, with four threads:

| simulations | time per move |
|---|---|
| 16 (Casual) | 55 ms |
| 64 (Club) | 221 ms |
| 256 (Strong) | 898 ms |
| 800 (Max) | 2,789 ms |

The top two levels are capped by a clock rather than a count, because a fixed number of simulations
is a fixed amount of *work* and not a fixed amount of *waiting* — and those differ by more than an
order of magnitude between a laptop and a phone. Max searches for up to two seconds and reports how
far it actually got, which on the machine above is about 560 of its 800 simulations.

Open `/bench/` on any device to measure it there.

**The rules are written three times now.** Twice in Python, cross-checked over 20,000 games a night and
frozen; once in TypeScript for the browser. The third one is held to the same standard: every
expectation it is tested against is *generated* from the frozen engine — 1,000 positions with the
exact discs each move flips, the input encoding, search visit counts, and whole games replayed move
for move. `reversi export-fixtures` writes them and CI fails if they drift.

That was not a formality. It caught a wrap-around bug on its first run: building the board mask
computed `(1 << 32) - 1`, but JavaScript takes shift counts modulo 32, so `1 << 32` is `1` and the
mask came out as zero. The engine kept working and quietly stopped believing in the bottom four
rows of the board.

## Configuration

Three profiles, layered on `configs/base.yaml`:

| Profile | Purpose | Board | Sims | Notes |
|---|---|---|---|---|
| `smoke4x4` | Pipeline validation only — **never a result** | 4×4 | 25 | Must learn in ≤ 10 min on CPU |
| `dev8x8` | Shake-out run before committing a night | 8×8 | 100 | ~60–90 min on one GPU |
| `full8x8` | The headline result | 8×8 | 300 | 6 workers; generations sized to ≤ 20 min so an 8-hour job ceiling costs little on preemption |

Override anything from the command line: `reversi train -c configs/full8x8.yaml --set mcts.n_simulations=200`.
Unknown keys are errors, not silent no-ops.

## Reproducibility

Every run writes, before doing any work:

```
runs/<run_id>/
  config.yaml   fully resolved configuration (hashed into every checkpoint)
  env.json      python / torch / CUDA / driver / OS / CPU / GPU / hostname / SLURM job
  git.json      commit, branch, dirty flag, diff sha256 (+ diff.patch when dirty)
  meta.json     run_id, seed, config sha256, config-hash history across resumes
  cmdline.txt   the exact argv that launched the run
```

`run_id` is `{timestamp}-{profile}-{git_short}-s{seed}`. A run that cannot write these does not start.

Each generation also writes `checkpoints/gen_NNNNN.pt` (weights, optimiser state, RNG state) beside
`gen_NNNNN.json` — the same metadata as plain text, plus the checkpoint's SHA-256. Resume walks
backwards from the newest generation until it finds one whose checksum still matches, so a process
killed partway through a write falls back one generation rather than loading a torn file. See
`docs/training.md`.

## Does it learn?

Yes — and the claim is measured by playing, never by pointing at a loss curve.

**8×8, after 120 generations of self-play from random weights** (72,000 games, 8 h 20 m on one
L40S). Bradley–Terry ratings fitted across a 78-pairing round robin of 7,800 games, anchored at
Random = 0:

| agent | Elo | 95% bootstrap interval |
|---|---|---|
| **generation 120** | **+1004** | [+949, +1070] |
| generation 35 | +862 | [+807, +928] |
| *0.0's generation 60, for comparison* | *+874* | *[+819, +938]* |
| generation 15 | +722 | [+670, +785] |
| generation 5 | +516 | [+469, +574] |
| Minimax-d4 (hand-written, depth-4 alpha-beta) | +482 | [+432, +544] |
| Greedy | +396 | [+336, +461] |
| Random | 0 | — |

The network 0.0 shipped is in the same fit deliberately. Its own tournament put it at +877 and this
one puts it at +874, but those two numbers are not comparable and the agreement is a coincidence:
**a rating is a coordinate the fit assigns given the field it measured, not a property of a
network.** Adding two generations to this very tournament moved every strong entrant by about 40
points without replaying a single game. What survives a change of field is the *difference* between
two entrants that both played the same opponents — which is why the claim is "+130 over what 0.0
shipped" rather than any single number above.

![Bradley-Terry ratings for thirteen agents with 95% bootstrap intervals. Generation 120 leads at
roughly +1004 with generations 100 and 114 alongside it and their intervals overlapping almost
entirely; the network 0.0 shipped sits near +874 between generations 35 and 65; minimax-d4 is near
+482 and Random is the zero point.](docs/figures/ratings.png)

*Everyone placed on one scale by fitting all 78 pairings at once, rather than chaining head-to-head
results — so the ordering does not depend on which matches happened to be played first. The bars for
generations 100, 114 and 120 overlap almost entirely; that overlap **is** the plateau, and it is why
generation 120 ships as the run's final checkpoint rather than as its best one.*

Generation 120's interval lies entirely above generation 35's, which lies entirely above generation
15's, which lies above generation 5's — the strict form of "later is stronger", three times over,
rather than point estimates that happen to be in the right order.

The baseline matters here: Minimax-d4 was *given* corner theory, mobility and frontier evaluation.
The agent was given none of it and had to find those ideas from its own games.

**Known limitations, stated rather than buried.** The agent plateaued — generations 100, 114 and 120
are statistically indistinguishable, their head-to-heads all spanning 50% — and it still scores less
against Greedy (80%) than against the stronger Minimax-d4 (95%), because self-play narrows its
training distribution away from the strange positions bad play produces. The non-transitivity is
smaller than it was at 0.0, where the gap was 63% against 100%, but it has not gone. Both are
written up in `docs/experiments.md`.

![Score against each baseline from generation 5 to 60. The line against minimax-d4 climbs from 0.57
to 1.00 while the line against Greedy falls from 0.90 to 0.63; the two cross around generation 20.](docs/figures/strength.png)

*The crossing lines are that second limitation, drawn. As the agent got better against the searcher
that plays well, it got **worse** against the one that plays badly — self-play stops producing the
strange positions Greedy walks into, so the agent stops practising them. Shaded bands are 95% Wilson
intervals; they are wide because each point is 30 games.*

<details><summary>4×4 pipeline validation (a fixture, not a result)</summary>

| | score over 200 colour-balanced games |
|---|---|
| 4×4 agent vs Random | **97.2%** |
| 4×4 agent vs Greedy | **93.2%** |

The agent scores near 100% as white and lower as black. That is not a lopsided agent: **white wins
4×4 Othello with perfect play**, which `tests/unit/test_solved_4x4.py` proves by solving the game
exactly (3,306 positions). As black it is defending a theoretically lost position. That test doubles
as an independent check on the rules engine — it never inspects a flip or a legal-move list, it just
plays every possible game to the end and asks who wins.

</details>

## The training loop

One generation: freeze the network, play a batch of games against itself, write those positions to a
shard, then take a few hundred optimisation steps sampling from a sliding window of recent games.

Two things the network learns from each position: what the search concluded (its visit distribution
over the legal moves), and how the game actually ended, seen from the point of view of whoever was to
move there. The search's answer is better than the network's own, which is what makes the loop climb.

Positions are stored as **bitboards, not encoded planes** — so changing the network's input format
does not invalidate games already collected. Every shard is checksummed, every write is atomic, and
a shard that no longer matches its checksum is dropped rather than trained on.

Measured on the dev laptop, `smoke4x4`: ~40s per generation, so the 12-generation profile lands
around 8 minutes on CPU.

![Two panels. Policy cross-entropy falls smoothly from 2.5 to 1.2 over 60 generations. Value mean
squared error drops sharply to 0.63 by generation 10, then drifts slowly back up to
0.66.](docs/figures/losses.png)

*Included as a diagnostic, and labelled as one on the figure itself. A falling policy loss means the
network is getting better at predicting what its own search will conclude — which is not the same
thing as playing better, and the arena above is what answers that. The value loss falling and then
drifting back up is expected rather than a bug: as play improves, games get closer, so guessing the
winner from a position genuinely gets harder.*

## Throughput

Self-play is where essentially all the compute goes — one 8×8 generation is tens of millions of
network calls against a few seconds of backpropagation. Measured on an RTX A1000 laptop GPU:

| | games/s | speedup |
|---|---|---|
| one game at a time | 0.072 | — |
| 48 games batched | 0.977 | **13.6×** |
| 6 worker processes | — | **2.25×** on top |

The reason batching matters so much is that a single position barely occupies a GPU at all — 499
positions/s at batch 1 versus 26,328 at batch 48, on the same card. It was never a hardware problem;
the GPU was being starved.

`bench/selfplay_bench.py` produces these numbers, and `bench/results/` holds them. Nothing in this
repo is optimised without a before/after measurement to point at.

![Self-play time per generation over 60 generations, holding between roughly 8.7 and 10
minutes.](docs/figures/throughput.png)

*About nine minutes per generation, and held there on purpose. The run had an eight-hour job limit,
so generations were sized so that being interrupted costs at most one of them — the alternative is
losing a night's work to a scheduler.*

## The rules engine

Implemented twice on purpose:

* `game/reference.py` — a grid of squares, walking outward in eight directions. Slow, and correct by
  inspection. This is the specification.
* `game/rules.py` + `game/bitboard.py` — two 64-bit integers per board, one bit per square. About ten
  times faster, and *not* correct by inspection.

Neither was written from the other; both were written from the rules of Othello. A test plays 20,000
random games through both and compares them at **every move** — legal moves, which discs each move
would flip, the resulting position bit for bit, and the final score. That agreement is the entire
correctness argument, and it exists because a rules bug does not crash: the AI would simply learn a
slightly different game and every measurement afterwards would be meaningless.

See `docs/how-the-engine-works.md`.

## Documentation

* `docs/web-app.md` — **running the site, deploying it, and what "there is no server" means**
* `docs/configuration.md` — every setting, what it does, and what breaks if you change it
* `docs/how-the-engine-works.md` — bitboards, passing, perspective, and the eight board rotations
* `docs/endgame.md` — **solving the last few moves exactly, and the puzzles built from it**
* `docs/training.md` — running a job, stopping one, resuming, and what a run leaves behind
* `docs/experiments.md` — one entry per run: hypothesis, config delta, outcome, decision
* `docs/roadmap-1.0.md` — what 1.0 adds, which experiments run and what they predict, and what gets cut first
* `docs/architecture.md` — layering, dependency rules, and the seven correctness contracts
* `docs/model_card.md` — training compute, data provenance, measured strength, limitations
* `docs/references.md` — which ideas came from papers, which from engine practice, and which from folklore
* `docs/decisions/` — one file per decision that would be expensive to revisit
  * `ADR-0001` — a board is two integers, and what that costs
  * `ADR-0002` — whose point of view a number is from (the value sign)
  * `ADR-0003` — symmetry, and the four rotations that cannot happen
  * `ADR-0004` — independent workers that write their own shards
  * `ADR-0005` — the agent runs in the browser, not on a server

## How this was built

Written with AI assistance, using Claude Code as a pair programmer for
implementation and review.

What that did not cover: the experimental standards this project is held to. The
two independent rules engines and the differential test between them, the seven
correctness contracts, refusing to cite a training loss as evidence of strength,
writing predictions down before a run rather than after it, and reporting the
plateau and the non-transitivity against Greedy instead of quietly omitting them
— those are the argument the repository is making, and `docs/` records the
reasoning behind each one.

## License

MIT
