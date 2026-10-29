# Step 3 — Does Training Against Fixed Opponents Transfer? The Pre-Registered Design

**Applies to:** Step 3 of the [experiment design proposal](experiment-design-proposal-2026-10.md), approved in scope at the late-October advisor meeting
**Prepared:** October 28, 2026
**Status:** Design, fixed before any training run starts. The held-out roster it evaluates on is frozen in [held-out-opponents.md](held-out-opponents.md); the two Step 2 follow-ups that shaped it are in [step2-failure-analysis.md](step2-failure-analysis.md) sections 7 and 8.

## 1. The question, and the one it is not

Steps 1 and 2 established what goes wrong: the self-play network is wiped out early by Greedy because self-play never produced positions where holding one disc was fatal, so the network rates them as wins. The obvious fix is to show it such positions, by letting it play Greedy during training. Step 3 is that experiment, with one twist that is the whole point: the result is read on opponents the network did *not* train against.

The question is: **does a small, fixed share of training games against Greedy and Minimax-d4 make the network better against opponents it has never seen, or only against those two?**

It is not "does playing Greedy fix the Greedy wipeouts". That is expected to be true and is reported as a seen-opponent result. A gain that appears only on the training opponents is the outcome the Go literature warns about (Wang et al. 2023; Tseng et al. 2025): defending against one attack buys resistance to that attack and nothing more.

## 2. Groups

All groups start from the same checkpoint and train for the same number of generations and gradient steps. One variable changes between A and B.

| group | training games per generation | what differs |
|---|---|---|
| **A, control** | 600 self-play games | nothing: the control recipe continued |
| **B, opponent mix** | 480 self-play games + 240 games against a fixed opponent | 20% of the training examples come from games against Greedy or Minimax-d4 |
| **D, signal only** | 600 self-play games | half of each value target is the position's disc share instead of the game's result; no external opponent |

Two training seeds per group. Earlier in the project, two seeds of one recipe differed by 48 rating points, so a single run per group would not separate the recipe from the seed.

**Starting checkpoint:** `control120`, the control recipe at generation 120 (`runs/control/checkpoints/gen_00120.pt`). It is the checkpoint with the cleanest version of the weakness: 72 wipeouts in 200 games against Greedy, unchanged from generation 60.

**Length:** 20 generations. The replay window holds 400,000 positions, about twelve generations of games, so after 20 generations every position the trainer samples was produced under the group's own recipe. The question is whether the weakness moves, not whether the network reaches a new peak; 120 generations from scratch would cost six times as much and answer the same question.

**Group C (Edax in the mix)** from the proposal is deferred to Phase 2. Edax does not wipe anyone out, so it has no bearing on the mechanism Step 2 found, and matching its search time against the other groups would add a confounder to a comparison that otherwise changes one thing.

**Group D** is the cheapest arm and the one that speaks to the self-play premise. If a better training signal fixes the wipeouts without any external opponent, the weakness was in what the value target told the network, not in self-play as such. The precedent is the ownership head: at the same generation, the network trained with it is wiped out a third to a half less often (Step 2, section 5). D asks the same question with a term aimed at the mechanism rather than at the whole board.

Its target is fixed here, before the run. Each position's value target becomes half the game's result and half the position's *disc share*, (own discs − opponent's discs) / (all discs) from the mover's view: `train.material_target_weight: 0.5` with `train.material_target: current`, in `configs/full8x8_step3_signal.yaml`, which is otherwise `configs/full8x8.yaml`. The share is near zero in a balanced position, so most targets barely move; it is about −0.95 where the mover holds one disc against forty, which is exactly where Step 2 found the value head confidently wrong, so a one-disc position that self-play labelled +1 is relabelled about 0. The blend is applied when a batch is drawn, so it covers the copied replay window from the first gradient step, not only the games D plays itself. The cost to watch is retention: a value head that is half material is half Greedy's judgement, and the retention and Edax rows are where that would show. The conventional alternative, blending in the *final* margin instead, is implemented beside it (`material_target: final`) but is not an arm: it leaves the sign of a one-disc position that went on to be won unchanged, so it does not address the mechanism.

D runs alongside A and B, one seed first; the measured cost on the cluster (section 6) leaves room for a second.

## 3. How the fixed opponents enter training

**A constant share, every generation.** Each generation of group B plays 480 ordinary self-play games and 240 games in which the network plays one side and a fixed opponent plays the other. The share never changes over the 20 generations. Concentrating the external games early would let them slide out of the replay window before the run ends; concentrating them late would leave the network no time to learn from them; a schedule that changed over time would be a second variable. A constant share keeps the composition of the replay window the same in every generation, which is what makes A and B differ in one thing.

**Which opponent, which colour.** The external games are spread evenly through the generation's game indices; the opponents take turns (Greedy, Minimax-d4, Greedy, ...) and the network's colour flips each time the list wraps around, so of 240 external games it plays each opponent 60 times as black and 60 times as white. All of this is a function of the game index alone, so a worker re-run or a resumed generation plays the same mix, as every self-play game already replays from its index.

**The network searches exactly as in self-play.** Same simulations per move (200), same exploration noise at the root, same temperature for the first 15 plies. The only change is who answers on the other side. The policy target for each of the network's moves is its own search's visit distribution, produced the same way as in self-play. The opponent's moves are not recorded: there is no search behind them to learn from, so they only advance the game.

**Why 20% of the examples.** Step 2 section 3 found that in self-play a side holds one or two discs after the opening about once per hundred games. Against Greedy the pilot produced such positions in a third of the games, and in training, with exploration noise on, more often. At 20% of examples, roughly a third of a generation's games are external, which puts hundreds of punished few-disc positions into the window per generation against the handful self-play supplies. The share is small enough to leave the training distribution mostly self-play, which is what "a small, controlled amount of opponent diversity" in the proposal asks for, and large enough not to be diluted to nothing in a 400,000-position window.

**Matched on examples, not games.** A self-play game yields a training example at every move with a real choice, from both sides. An external game yields examples only from the network's moves, about half as many. 480 self-play games plus 240 external games therefore produce about as many examples as the control's 600 self-play games, and the number of gradient steps per generation is fixed by the config (800 steps of 1,024 positions) in every group. Game counts, example counts and steps are logged per generation and reported.

**Both heads learn from the external games.** Each recorded position carries the game's final result as its value target, as in self-play. The alternative, using external games for the policy target only, was considered because of the overconfidence risk the proposal raised: the network wins most games against Greedy, so positions from those games are labelled "win" more often than the same positions would be in self-play. The Step 2 head probe (section 7 of the failure analysis) decides against it. The value head has the sign backwards one ply before a wipeout, and the policy head gives the wiping reply a few percent; at fifty simulations the prior is what lets the loss through, but the policy head is trained only on positions where the network is to move, so an external game produces no policy target at Greedy's node at all. The value target is the lever a mixed-opponent game pulls: a one-disc position labelled with the loss that followed is exactly the example the replay window never contained. Value calibration is checked after training in both groups: the network's value estimate against the actual outcome, on self-play games and on the held-out games, binned by estimate. A group whose estimates drift from the outcomes has paid for its opponent exposure with overconfidence, and that is reported as a cost.

## 4. Evaluation

Every group's final checkpoint, and the starting checkpoint, under the frozen protocol of Step 1: 200 games per pairing, 100 seeded 4-ply openings each played with both colours, 50 simulations per move, no exploration noise, Wilson intervals per pairing, wipeouts counted per pairing. The scorecard is the one in the roster document, section 6:

| row | opponents | what a change there means |
|---|---|---|
| **held out** (primary) | Edax at a level not used in training; the random-playout search at its calibrated budget; any secondary roster opponent obtained in time | the transfer the experiment exists to measure, one row per opponent, never averaged |
| **retention** | a checkpoint from another training run; the strongest Edax level the network can be measured on | whether general strength was traded for the gain |
| **seen** | Greedy, Minimax-d4 | the expected gain; reported, not counted as transfer |
| **seen family** | greedy-corner, greedy-d2, noisy-greedy, corner, mobility | the same ideas as the training opponents in other clothes; transfer within a family |
| **sanity** | Random | nothing is broken |

Wipeout counts are reported for every row. The held-out and seen-family rows also carry the Step 2 descriptors: median searched value at the network's last move in each wipeout, and whether a safe move existed.

## 5. Predictions, written before the runs

1. Group B's wipeouts against Greedy fall sharply, to a handful per 200 games; group A's do not move. This is the seen-opponent result and is not evidence of transfer.
2. Against the seen-family opponents, B improves in proportion to how closely each resembles Greedy: greedy-corner and noisy-greedy most, corner and mobility least.
3. The held-out rows are the open question, and the three readable outcomes are:
   - **B improves on held-out rows and holds retention:** opponent exposure taught something general about low disc counts, not just Greedy's habits.
   - **B improves only on seen and seen-family rows:** the network learned its two training opponents. This is the Go-literature outcome, and it is a clean negative result.
   - **B improves on seen rows and loses retention:** the exposure cost general strength. Reported as such; it would argue for a smaller share or for group D's approach.
4. Group D is predicted to cut wipeouts against Greedy without external games, by about the amount the ownership head did (a third to a half). Its retention row is predicted to fall by a measurable amount, since half of the value head's target is now material; a fall larger than B's would mean the signal-only route is the costlier one. Whether it moves the held-out rows is not predicted.
5. The head probe (`scripts/probe_heads.py`) is re-run on each group's final checkpoint, on whatever wipeouts remain. For group B the raw value at the fatal position is predicted to fall well below the +0.99 of the starting checkpoint, while the prior on Greedy's wiping reply is predicted to change little, because nothing in the design trains the policy head at the opponent's node. If the value stays at +0.99 and the wipeouts fall anyway, the mechanism is not the one Step 2 identified, and that is reported.

## 6. Cost

The control run's generation took about 12 minutes on a laptop GPU with 6 self-play workers, and the run itself used 22 workers on the cluster (`runs/control/cmdline.txt`). Twenty generations is therefore 4 to 8 GPU-hours per run depending on the machine; four runs, 16 to 32 GPU-hours; six with group D, 24 to 48. Group B's external games cost CPU time for Minimax-d4's search (milliseconds a move) and nothing on the GPU. Evaluation afterwards is CPU only: the Step 1 pilot's 3,000 games took 47 minutes on a laptop, and the scorecard here is of the same order per checkpoint.

## 7. How a run is launched

The training loop resumes a run directory from its newest checkpoint and accepts a changed config with a warning, recording the old and new config hashes in the run's `meta.json`. A fine-tune from `control120` is therefore a copy of the control run's directory, continued under the group's config:

```bash
cp -r runs/control runs/step3-A-s1            # checkpoints up to gen_00120 and the replay window
reversi train -c configs/full8x8.yaml --run-id step3-A-s1 --generations 140 --device cuda     --set selfplay.n_workers=22

cp -r runs/control runs/step3-B-s1
reversi train -c configs/full8x8_step3_mix.yaml --run-id step3-B-s1 --generations 140 --device cuda     --set selfplay.n_workers=22

cp -r runs/control runs/step3-D-s1
reversi train -c configs/full8x8_step3_signal.yaml --run-id step3-D-s1 --generations 140 --device cuda     --set selfplay.n_workers=22
```

The second seed of each group adds `--set seed=1338`. On the cluster the same runs go through `slurm/submit_chain.sh`, with the seed passed as `EXTRA_ARGS="--set seed=1338"`. Copying the replay window as well as the checkpoints matters: both groups then start from the same 400,000 positions and the same weights, and diverge only through the games they play from generation 121 on. `configs/full8x8_step3_mix.yaml` is `configs/full8x8.yaml` with the `selfplay` block changed and nothing else; the two files are kept side by side so the diff is the experiment.

## 8. What is logged, per group and generation

Games played by kind, the network's score and wipeout rate in the external games, training examples, gradient steps, the replay window's size, the random seeds and wall time, all in the run's `metrics/` files as for any run. Which opponent and colour each external game had follows from its index. The evaluation reports carry the entrant specifications and the git commit, as every arena report does.

**Gate C**, unchanged from the proposal: advance to a write-up only if the training ran as specified and either shows a clear held-out improvement without unacceptable regression, or a controlled negative result. No claim about adaptive adversaries follows from any of this; the fixed opponents are fixed.

## References

- Wang et al. (2023), *Adversarial Policies Beat Superhuman Go AIs*. https://proceedings.mlr.press/v202/wang23g.html
- Tseng et al. (2025), *Can Go AIs Be Adversarially Robust?* https://arxiv.org/abs/2406.12843
- Czempin and Gleave (2022), *Reducing Exploitability with Population Based Training*. https://arxiv.org/abs/2208.05083
