# Roadmap: reversi-zero 1.0

## Where 0.0 landed

One 8×8 training run: a network of 6 residual blocks × 64 channels (about 459k parameters), 60
generations, 36,000 self-play games, 9.1 hours on a laptop GPU. The final agent rates +877 Elo over
random play, beat depth-4 minimax 30–0, and is statistically even with the Edax engine at level 5. A
static browser app runs the network in WebAssembly with four rated opponents and a measured four-level
difficulty ladder. Seven correctness contracts (C1–C7 in `architecture.md`) are enforced by tests. CI
runs on CPU only.

## What 1.0 is for

Two goals, in this order:

1. **A stronger agent, trained on a SLURM GPU cluster, using ideas from the literature.** Run 2
   ended with the conclusion "the next test is capacity, not weighting". A cluster makes bigger
   networks and more experiments affordable.
2. **A UI that reads as a game, not a demo.** A visual identity, a hub page that presents a
   collection of agents, and game features: move history, replay, a win-rate graph, records.

Plus one structural change: **a `Game` interface**, so that Reversi becomes one implementation and
Gomoku (五子棋) can be added in 1.1 as a new module rather than a rewrite. No second game ships in 1.0.
The seam is proved by a tiny game that exists only in the tests.

### Constraints

| Constraint | Value | Consequence |
|---|---|---|
| Time | 4 weeks × ~2 h/day ≈ 56 h of hands-on work | Cluster jobs run unattended and do not count. Human hours are the bottleneck; see the cut order. |
| Compute | SLURM GPU cluster. GPU model, wall-time limit, GPU-hour quota, and node internet access are not yet known | Day 1 is fact-finding plus a 4×4 smoke job. The plan assumes 24-hour jobs chained through the existing resume logic. |
| Hosting | Static site, no backend, no accounts | A bigger network needs WebGPU inference in the browser, with WebAssembly as the fallback. Records live in browser storage. |
| Quality gates | `ruff check`, `ruff format --check`, `pyright`, `pytest -m "not slow and not gpu"` | Every pull request is green on all four. |

### Debt that 1.0 closes

All of this is verified in the code as of the 0.0 release.

- `reversi arena` and `reversi bench` are stubs that exit with code 2 (`src/reversi/cli.py`).
- The training loop never evaluates itself. The whole `arena` block in the config is read by nothing.
- `docs/crossgen.json`, which the README figures and the web app's opponent list both read, cannot be
  regenerated from any command in the repository.
- No SLURM support (`training.md`, "On a job scheduler").
- The easy difficulty levels are inconsistent, and that is unmeasured (`experiments.md` lists three
  fixes "for 1.0").
- The README marks `serve` and `calibrate` as planned although both work. Several docstrings still
  describe features as "missing until day 7" that have existed since day 7.

---

## 1. Scope

### Main scope

| # | Workstream | Hours |
|---|---|---|
| A | Cluster bring-up, SLURM scripts, experiments E1 (capacity), a control run, E2a (ownership head), a `c_puct` sweep | 12 |
| B | Evaluation tooling: `reversi arena`, in-loop quick evaluation, `best.pt` | 9 |
| C | `Game` interface, steps G1–G4, plus the seam test | 11 |
| D | Difficulty consistency: measure spread, compare weakening methods, rate the offered combinations, recalibrate on the new network | 4 |
| E | Web 1.0: hub and directory move, WebGPU path, visual identity, replay and win-rate chart, records, manifest v2 | 20 |
| F | Docs, hygiene, release: experiment entries, model card, README, stale docstrings, the `v1.0.0` tag | 4 |
| | **Total** | **~60** |

Sixty hours against fifty-six is deliberately tight. Section 6 says what is cut first.

### Stretch, only if ahead

E2b playout cap randomisation · E3 Gumbel root selection · `reversi bench` · a hint button ·
save and resume of an in-progress game · a mobile bottom-sheet layout · a self-hosted font · a full
sound set.

### Out of scope for 1.0

Gomoku or any second playable game (1.1) · a backend or accounts · a leaderboard · multi-GPU · an
endgame solver · beating Edax at full strength · making the TypeScript search generic (1.1, with Gomoku).

---

## 2. The research shortlist

Each experiment is **pre-registered**: its prediction is written into `experiments.md` before the job
is submitted. Training loss is never cited as evidence of strength.

### E1 — Capacity

A bigger network, nothing else changed. This is run 2's own conclusion, and it is a config-only change,
so it is the first cluster job.

| Network | Parameters | Compute per position, relative to 6×64 |
|---|---|---|
| 6×64 (run 1) | 458,696 | 1× |
| 8×96 (fallback) | ~1.34 M | ~3× |
| **10×128** | **~2.97 M** | **~6.7×** |

Prediction: at matched generation (40 and 60), E1's Bradley–Terry interval lies entirely above run
1's, and the value loss drops below 0.60 by generation 30 (run 1 never went under 0.64). If strength
rises but value loss does not, capacity helped the policy only.

A **control** run of the unchanged `full8x8.yaml` is submitted alongside. It costs no coding, gives a
hardware-matched wall-clock baseline, and is a second seed of run 1, which the experiments log flags as
a single-seed result.

### E2 — An auxiliary ownership head

Source: D. J. Wu, *Accelerating Self-Play Learning in Go*, arXiv:1902.10565 (the KataGo paper).

Today the value head learns from one number per position: who won. An **ownership head** also
predicts, for each of the 64 squares, who owns it when the game ends (+1 mine, −1 theirs, 0 empty, from
the point of view of the player to move). That is 64 extra training signals per position about "who is
winning where", which is exactly what a single win/loss reward lacks. Unlike run 2's heavier value
weight, it does not compete with the policy for the same scalar. The weight is 1.0: the same scale as
the value loss, since both are mean squared error on tanh outputs, and run 2 showed that 4× on the
value term starved the policy.

Prediction: value loss below 0.62 by generation 30, and strength at generation 40 with an interval
above run 1's.

### E2b (stretch) — Playout cap randomisation

Same paper. Most self-play moves use a cheap search (80 simulations); a random 25% use a full one
(400). Only full-search moves produce policy targets; every move produces value and ownership targets.
The point: a value target only needs the game to be played out sensibly, so most moves can be cheap,
and the policy targets that are kept come from a much deeper search than could be afforded on every
move. Roughly doubles useful games per GPU-hour. Touches contract C4, so it is the riskier half and is
scheduled after E2a.

### E3 (stretch) — Gumbel root selection

Source: Danihelka, Guez, Schrittwieser, Silver, *Policy improvement by planning with Gumbel*, ICLR 2022.

At the root of the search, add Gumbel noise to the network's move scores, keep the top *m*, then spend
the simulation budget in halving rounds: visit every survivor equally, drop the worse half, repeat. At
16 simulations, ordinary PUCT mostly reflects the prior plus noise; Gumbel still makes a principled
comparison among the moves it did look at. That is why it is the candidate fix for the inconsistent
easy levels. Root-only in 1.0; using it to produce training targets is out of scope.

### The `c_puct` sweep

`c_puct` is the constant that trades off exploring new moves against exploiting good ones during
search. It has never been tuned (`references.md` calls this "an obvious thing to test"). Play-time
only, no training: `reversi arena` on `gen_00060.pt` at c_puct ∈ {1.0, 1.5, 2.5, 4.0}, 100 games each,
overnight on a laptop CPU in week 1. A clear winner is used for E2 self-play and recorded.

One line for the model card: Othello was weakly solved as a draw in 2023 (H. Takizawa, *Othello is
Solved*, arXiv:2310.19387). Perfect play exists; this project is not chasing it.

---

## 3. Workstreams

### A. Cluster and experiments

New directory `slurm/`:

```
slurm/env.sh           module loads; uv sync --extra cu124 --extra obs; export RZ_RUN_ROOT
slurm/smoke4x4.sbatch  day-1 test: 1 GPU, 30 min, reversi train -c configs/smoke4x4.yaml
slurm/bench.sbatch     bench/selfplay_bench.py on a node -> bench/results/cluster-<gpu>.json
slurm/train.sbatch     the real job
slurm/cpu.sbatch       CPU partition: any CPU-only command, e.g. reversi arena --suite crossgen
slurm/submit_chain.sh  submits K copies of train.sbatch with --dependency=afterany
slurm/fetch_run.sh     rsync checkpoints/{latest,best}.*, metrics/, arena/, config.yaml, meta.json
slurm/README.md        how to submit, read logs, and resume by hand
```

`train.sbatch` asks for one GPU, 16 CPU cores, 24 hours, `--signal=B:USR1@900` and `--requeue`, and
runs `reversi train -c $CONFIG --run-id $RUN_ID --generations $TOTAL --device cuda`. Everything the
script relies on already exists: `obs/signals.py` treats USR1 as "finish this generation and stop",
and `run_training` treats `--generations` as a total and exits in seconds when the run is complete.
So a long run is a chain of identical 24-hour jobs, each resuming where the last stopped. Nothing under
`src/` changes. `SLURM_RESTART_COUNT` already lands in `env.json`, which is the evidence criterion S9
asked for.

Day 1, in order: record the cluster facts in `training.md`; `uv sync --extra cu124` on the login node;
submit the smoke job, then resubmit the same run with a 10-minute limit to force the
signal → clean stop → resume path; submit the bench and read its inference rate at batch 24 to decide
between 10×128 and the 8×96 fallback (a generation must stay under about 20 minutes); submit E1 and
the control.

Configs: `configs/full8x8_e1_10x128.yaml`, `configs/full8x8_e1_8x96.yaml`,
`configs/full8x8_e2_ownership.yaml`. 120 generations each, chained 4 × 24 h.

### B. Evaluation tooling

**`reversi arena`.** The parallel round-robin driver that already exists inside
`difficulty/calibrate.py` is generalised into `arena/entrants.py` (a picklable description of one
contestant, and a function that builds the agent from it) and `arena/tournament.py` (a process pool,
one thread per worker, on the CPU). Suites, each writing under `runs/<id>/arena/`:

- `baselines`: one checkpoint against the configured baselines.
- `crossgen`: the checkpoints the pruner keeps, plus baselines. This **regenerates `crossgen.json`**.
- `final`: `crossgen` plus Edax level 5 when Edax is installed.
- `combos`: every level × opponent combination the web app offers, each against three baselines and
  its own generation's Strong level, 60 games each, one Bradley–Terry fit (see D).

**Quick evaluation inside training.** Every `arena.every_n_generations`, the newest weights play 40
games at 50 simulations against random, greedy, and depth-2 minimax. Baseline-versus-baseline pairings
are played once per run and cached. The agent's Bradley–Terry rating from that small table is written
into the checkpoint sidecar as `elo_estimate`, and `best.pt` is updated when it improves. Cost is about
five minutes per five generations, roughly 10% overhead, accepted so the loop stays synchronous
(ADR-0004). Stated plainly in the docs: `elo_estimate` is a within-run, low-precision curve for
choosing `best`. It is not the cross-generation rating, and the two scales agree only on the anchor.

### C. The `Game` interface

New `src/reversi/game/protocol.py`, standard library and numpy only, never torch. It names what the
rest of the pipeline needs from a game: identity and shape (`board_size`, `policy_size`, `pass_action`,
`in_planes`, `max_plies`); rules (`initial_state`, `legal_actions`, `apply`, `is_terminal`, `result`);
the network interface (`encode`, contract C1); storage (`to_bitboards` / `from_bitboards`, so shards
stay "two bitboards and a side to move", a stated 1.0 limitation); symmetry (`symmetries`,
`transform_state`, `policy_gather`, contract C6, where special actions map to themselves); and an
optional `ownership` target for E2.

`ReversiGame` delegates to the existing `rules`, `scoring`, `symmetry`, and `features.encode`. The
Reversi `State` class is unchanged and opaque outside `game/`. The network gains an explicit
`policy_size`, and its architecture record gains `game` and `ownership` fields, with a normaliser that
fills defaults so run 1 and run 2 checkpoints still load and resume.

Order, each step a pull request green on all four gates, Reversi behaviour byte-identical:

- **G1** protocol, `ReversiGame`, registry, no callers. Test: agrees with `rules.*` on 2,000 random positions.
- **G2** the search takes a game. `test_mcts.py` passes unchanged.
- **G3** self-play and data take a game. Gates: batched self-play still produces identical games to
  unbatched, and a shard written on a fixed seed has the same sha256 before and after.
- **G4** network, checkpoints, training loop, config. Gates: resume, ONNX export, and a new test that
  the released `reversi-8x8-gen60.pt` still opens.
- **Seam test** a 3×3 three-in-a-row game under `tests/support/`, with 2 input planes, 9 actions and no
  pass. Three generations of training produce width-9 targets; augmentation gives batches of shape
  (B, 2, 3, 3); a won-in-one position gets a root value of at least 0.95 (contract C2, stated
  game-agnostically). `ADR-0006` records the interface and its storage limitation. The contracts in
  `architecture.md` are restated in `Game` terms.

Making the arena, agents and difficulty code generic (G5) is stretch. In 1.0 they stay Reversi-typed,
and the ADR says so.

### D. Difficulty consistency

This closes the three items in `experiments.md`, "Reported from play: the easy levels feel inconsistent".

1. **Measure spread.** The same pairing in 15 blocks of 20 games, reporting the distribution of block
   scores and the full distribution of per-move value drops, into `difficulty_spread.json`.
   Prediction: Strong and Max show only binomial noise; Casual and Club show more.
2. **Rate what the interface offers.** The `combos` suite above, one overnight on 8 CPU workers. If it
   is cut, the interface hides levels for generations that were never rated, and says so.
3. **Compare weakening methods.** Three ladders on one checkpoint: (a) the current mix of simulations,
   temperature, top-k and guardrail; (b) simulations only, deterministic; (c) Gumbel, if E3 lands.
   Judged on criterion S15 (gaps of at least 80 Elo, disjoint intervals) **and** on spread. The ladder
   that passes S15 with the least spread ships. The chosen constants are mirrored into the TypeScript
   `levels.ts`; Python and TypeScript must match.
4. **Recalibrate on the 1.0 network** with `reversi calibrate --write-config`, about six CPU hours unattended.

### E. Web 1.0

**Directory move and hub, first.** Vite multi-entry pages, no router library, Cloudflare's `404-page`
handling kept so a missing `.onnx` is still a 404. URLs: `/` hub, `/reversi/` game, `/bench/` (exists),
`/gomoku/` reserved for 1.1. Layout: `web/src/hub/`, `web/src/shared/` (engine session, worker wrapper,
move choice, storage, theme, sound, and every UI component that is not the board), and
`web/src/games/reversi/` (engine, state, ui). Fixture files move but their contents do not, so the CI
diff against the Python-generated fixtures keeps working. A `GameScreenModule` contract (rules, levels,
manifest, engine factory, board component, move encoding) is what Gomoku implements in 1.1. The hub
shows a Reversi card with the opponent count, the strongest rating, and the visitor's record, and a
muted card for Gomoku marked as coming in 1.1. The TypeScript search stays Reversi-specific in 1.0,
because its visit counts must match Python count for count.

**WebGPU path.** The session loader imports either `onnxruntime-web/webgpu` or `onnxruntime-web/wasm`
dynamically, so a visitor who gets the CPU path never downloads the GPU build. Backend choice is a pure
function of the visitor's preference, whether a GPU adapter exists, and the network's size, so today's
small networks keep the measured WebAssembly path and only a network of 96 channels or more asks for
the GPU. Fallback chain: create the GPU session, run one warm-up evaluation (shader compilation, and it
catches adapters that pass detection but fail on first use), and on any error create a CPU session from
the same module, then the existing threaded-then-single-thread logic. The backend in use is reported in
the per-move line. The bench page measures both. CI's headless browser has no GPU adapter, so CI
exercises exactly the fallback. Why this is needed: at 10×128 the WebAssembly cost is about 20 ms per
simulation, so the Strong level's 1.2-second cap would fit only about 60 simulations and the top of the
ladder would collapse into Club.

**Visual identity.** The board stays a grid of DOM buttons: keyboard navigation, screen-reader labels
and every end-to-end selector come free. A frame drawn with CSS gradients, star points, and a–h / 1–8
labels. The disc flip becomes a real 3D turn with two faces, rippling outward from the placed disc at
60 ms per ring of distance. A placement scale-in. A manual light/dark toggle on top of the existing
theme tokens. Sound synthesised with Web Audio, no files, with a persisted mute. A game-over dialog
with the result, the counts, the win-rate sparkline, the record against this opponent, and Play again /
Swap sides / Review game / Back to games.

**Game features.** Move history with replay and a scrubber, built on the immutable per-ply history the
reducer already keeps; browsing history does not cancel a search in progress. A win-rate chart as an
inline SVG over `winProbabilityHistory`, which the state module already computes. Records and streaks
per opponent and level in versioned browser storage, with corrupt values quarantined rather than
crashing the page. Stretch: a hint, save and resume of an in-progress game, a mobile layout pass.

**Manifest v2.** The Python manifest generator emits a version, the game, the release tag, and per
model the run, the architecture and the checksum from the export sidecar. Filenames carry the run so
two runs' generation 60 cannot collide. The rule that a generation which was never rated cannot appear
in the opponent list is preserved structurally. Run 1's generation 60 enters the new round robin so it
appears on the same scale as the 0.0 network. Publishing the new model release is a release decision,
taken explicitly.

### F. Docs, hygiene, release

- `experiments.md`: E1, control and E2a entries, predictions first, results after; the `c_puct` sweep;
  spread and the ladder comparison; the combinations table. `model_card.md` for the 1.0 checkpoint.
  `training.md`'s scheduler section replaced with what was verified on the cluster. README status,
  figures regenerated, stale "(planned)" markers removed.
- Stale docstrings in `train/loop.py`, `arena/match.py`, `nn/model.py`, `ckpt/manager.py` and
  `train/trainer.py` corrected. Dead config fields either wired (`arena.*`) or removed
  (`train.checkpoint_every_steps`, `obs.resource_sample_seconds`, `obs.diagnostic_positions`, `obs.wandb.*`).
- Version 1.0.0 in `pyproject.toml` and `web/package.json`. Export, ONNX export, and fixture export on
  the chosen checkpoint: the fixtures must be byte-identical if the `Game` refactor is correct, which
  is a free regression check. The `v1.0.0` tag and release are explicit release decisions.

---

## 4. Week by week, about 14 hours each

**Week 1 — cluster up, jobs running, evaluation tooling.** Days 1–2: cluster facts, `slurm/`, smoke,
forced-resume check, bench, E1 and control configs and predictions, submit E1 and the control.
Days 3–5: `arena/entrants.py`, the generalised round robin, the arena suites, the CLI test update; the
`c_puct` sweep overnight. Days 6–7: quick evaluation, sidecar update, `best.pt`, the `arena` metrics stream.

**Week 2 — E2 code, G1–G2, web foundations.** Days 8–10: the ownership target end to end, the local
4×4 gate, a cluster smoke, submit E2a. Days 11–12: the web directory move, hub page, CI path edits.
Days 13–14: G1 and G2; fetch E1 and control results as they finish and run `crossgen` on them.

**Week 3 — G3–G4, WebGPU, visual identity.** Days 15–16: G3 and G4. Days 17–18: the WebGPU path and
bench. Days 19–21: board frame, 3D flip, placement animation, theme toggle, placement sound; submit the
spread and ladder jobs; write the E1 and control entries.

**Week 4 — features, seam proof, the checkpoint, release.** Days 22–23: move history, replay, the
win-rate chart, the game-over dialog, records. Day 24: the seam test, ADR-0006, the architecture doc.
Days 25–26: the E2a entry; choose the 1.0 checkpoint (best interval across E1, E2a and control); the
`combos` job; choose the ladder; recalibrate; manifest v2; export and fixture check. Days 27–28: model
card, README, `training.md`, stale docstrings, version bump, final gates.

If the 1.0 network is late, the web work ships against the run 1 networks, and the new network is a
manifest swap plus one bench run.

---

## 5. Verification

| Item | Must still pass | New evidence |
|---|---|---|
| Every PR | `ruff check`, `ruff format --check`, `pyright`, `pytest -m "not slow and not gpu"` under 90 s | |
| SLURM | (CI is CPU-only) | the smoke job's log, and an `env.json` with `SLURM_RESTART_COUNT ≥ 1` committed as `docs/slurm_smoke_evidence.json` |
| E1, control, E2a | the 4×4 learning gate with the new config knobs | pre-registered predictions in `experiments.md`; a `crossgen.json` per run with intervals |
| `reversi arena` | `test_cli.py`, with `arena` removed from the stub list | a tiny-run `arena --suite baselines` produces ratings |
| Quick evaluation | `test_loop.py`, `test_resume.py`, the gate's time budget | sidecars carry `elo_estimate`; `best.pt` is the maximum; the `arena` metrics stream has rows; cached baselines survive a resume |
| Ownership | `test_onnx_export.py` (still exactly two outputs), `test_replay.py` | ownership permutes with the board under every symmetry; old shards load with the target marked absent; the loss ignores absent rows |
| `Game` G1–G4 | `test_mcts.py`, `test_game_batch.py` (batched equals unbatched), `test_web_fixtures.py` (byte-identical fixtures), the property tests, API import isolation, the architecture-mismatch refusal on resume | `ReversiGame` agrees with `rules.*` on random positions; the released export still opens; the seam test |
| Difficulty | `test_calibrate.py` | `difficulty_spread.json`, the ladder comparison, `difficulty_combos.json`, a regenerated `difficulty_calibration.json` |
| Web | the Vitest suites at their new paths, the end-to-end game on `/reversi/`, the CI fixture diff | new unit tests for the reducer's view and play-from actions, the records aggregates, the backend choice, the hub registry; new end-to-end tests for the hub, replay, records and theme; one test asserting the WebAssembly fallback runs in CI |
| Release | a clean clone passes `make test` and `npm run build` | `reversi --version` prints 1.0.0; bench numbers for both backends on a laptop and a phone in `web-app.md` |

---

## 6. Cut order if behind

Whole items only, never half-finished ones.

1. E3 Gumbel (already stretch)
2. E2b playout cap randomisation (stretch)
3. `reversi bench` (stretch)
4. The `combos` suite. Fallback: hide unrated generations in the interface, documented.
5. Records and streaks. Fallback: a session-only tally in the game-over dialog.
6. The win-rate chart. The meter remains.
7. Sound beyond one click; the theme toggle.
8. The seam test. G1–G4 still land, and the ADR states the seam is exercised only by Reversi.
9. Quick evaluation inside training. Fallback: `arena --suite crossgen` after the fact only.

**Never cut:** E1, the control and E2a (they cost cluster time, not hours); the SLURM scripts;
`reversi arena crossgen` (nothing else can regenerate a file three consumers read); G1–G4; the hub and
directory move (structural for 1.1); the WebGPU path (required by the bigger network); the board, flip
and game-over dialog; move history and replay; manifest v2.

---

## 7. Risks

- **Cluster facts.** The GPU-hour quota decides whether E2a and E2b both run. The wall-time limit
  decides chain length. No internet on compute nodes means the environment is built on the login node.
- **10×128 may be too slow per generation**, because tree search is Python and CPU-bound and the
  bigger network's cost may not hide behind it. The day-1 bench decides; 8×96 is the ready fallback.
- **WebGPU availability.** Chromium yes; Safari 26 and later; Firefox partial. The fallback is the
  common path outside Chromium, and it is what CI tests. GPU latency for one position of a 10×128
  network is a guess until measured.
- **The difficulty ladder must be recalibrated** for the new network, and the constants mirrored in
  TypeScript.
- **The scope is tight.** The stretch list is real, and none of it starts before week 4's main items
  are green.
