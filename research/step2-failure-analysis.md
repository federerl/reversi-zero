# Step 2 — Why the Network Loses to Greedy

**Applies to:** Step 2 of the [experiment design proposal](experiment-design-proposal-2026-10.md), following the [Step 1 pilot](step1-pilot-results.md)
**Date:** October 25, 2026
**Data:** the pilot's recorded games (`runs/research/pilot-2026-10-games/`, not committed), Run 1's self-play replay window (`runs/20260827-030939-full8x8-60cfdda-s1337/replay/`, not committed), replays of the three Greedy pairings at 256 and 800 simulations per move, and the pilot protocol played by a second seed of the ownership recipe (reports under [`results/step2/`](results/step2/))
**Scripts:** [`analyze_wipeouts.py`](../scripts/analyze_wipeouts.py), [`scan_replay_disc_counts.py`](../scripts/scan_replay_disc_counts.py), [`probe_heads.py`](../scripts/probe_heads.py) (section 7); `analyze_wipeouts.py --opponent` and `analyze_pilot.py` read any fixed opponent's games (section 8)

**Gate B: a repeatable failure pattern with a supported mechanism.** The network is wiped out by Greedy because it rates a position where it holds one or two discs as a near-certain win, the search mostly agrees, and both are doing what the training data taught: in self-play, holding very few discs after the opening is rare and, when it happens, the side holding them usually wins. More search reduces the losses but does not change their character.

## 1. What the network believes while it is being wiped out

Every recorded network move carries the network's value estimate for the position before search (the *raw network value*) and after the tree search (the *searched value*), both from the network's point of view on a scale where +1 means "I win". The 177 wipeouts from the pilot, lined up by the network's distance from its last move:

| network | wipeouts | at its last move: discs held | searched value (median) | still positive | raw network value (median) |
|---|---:|---:|---:|---:|---:|
| gen60 | 70 | 1 | +0.91 | 96% | +0.99 |
| control120 | 72 | 1 | +0.95 | 100% | +0.99 |
| gen120 | 35 | 1 | +0.84 | 94% | +0.98 |

In 172 of 177 wipeouts the searched value was still positive at the network's last move, one ply before it had no discs left. In no game did it turn negative more than one move before the end. The raw network value is more confident than the searched one, so search is pulling the estimate in the right direction, just not far enough.

![Searched value and disc count over the network's last ten moves in every wipeout](figures/wipeout-values.png)

The lower panels show the shape of these games: for ten moves the network holds one to four discs, and the upper panels show its estimate climbing toward +1 as the count falls. The network is not blind to its disc count; it reads a low count as *good*.

**The won games look the same.** At the network's fifth move after the opening, the median searched value against Greedy is +0.91 in games it went on to win and +0.93 in games it was wiped out in. The network cannot tell the two situations apart, because to it they are the same situation: few own discs, many opponent discs, plenty of empty squares.

## 2. The losses were avoidable, and the search saw the escape

Replaying each wipeout with the rules module and searching exhaustively from the network's last few positions:

| network | wipeouts | a safe move existed at the last move | at the second-to-last | at the third-to-last |
|---|---:|---:|---:|---:|
| gen60 | 70 | 68 | 70 | 70 |
| control120 | 72 | 70 | 72 | 72 |
| gen120 | 35 | 31 | 35 | 33 of 33 |

"Safe" means a move after which no reply Greedy might choose (it plays any of its maximum-flip moves, breaking ties at random) leaves the network with zero discs. In 169 of 177 wipeouts such a move existed one ply before the end; one move earlier, in every one.

The search did not miss these moves. At the network's last move, the safe moves received a median 24% (gen60), 34% (control120) and 46% (gen120) of the search's visits, and in two thirds of the games a safe move was the second-most-visited move. The search compared the safe move with the fatal one and preferred the fatal one. For that to happen, the subtree under the fatal move must not have found Greedy's wiping reply, or found it too rarely to outweigh the network's +0.99 for everything else in that subtree. At 50 simulations spread over several candidate moves and their replies, a reply the policy head assigns a small prior may never be visited.

## 3. The training data said this was fine

Run 1's replay window, the positions the trainer was learning from over generations 31 to 60: 1,040,129 positions from about 18,000 self-play games. The first eight plies are excluded, because every game passes through tiny disc counts there by rule (after black's first move white holds exactly one disc).

| a side holds at most | positions from ply 8 on | per game | when that side was to move, it went on to win |
|---:|---:|---:|---:|
| 1 disc | 21 | 0.001 | 81% |
| 2 discs | 183 | 0.010 | 74% |
| 3 discs | 3,568 | 0.198 | 56% |
| 5 discs | 38,268 | 2.1 | 56% |

Two facts, and together they are the mechanism:

- **The position is almost absent from training.** A side holding one or two discs after the opening occurs about once per hundred self-play games. The pilot produced it in a third of the games against Greedy.
- **When it does occur, the side holding few discs usually wins.** 74% to 81% of the time. In self-play both players are the same network, and neither has any inclination to flip every disc it can, so a low disc count never gets punished. It correlates instead with what it correlates with in good Othello: few discs early means more mobility and fewer targets, and the side with few discs is often the side that is winning.

So the network learned a rule that is true about the games it was trained on and true in strong play: holding few discs in the opening is good. What it never saw is the boundary of that rule. Against an opponent that flips as many discs as it can, one disc is not a strong position, it is one move from the end, and nothing in a million training positions says so.

This is the self-play distribution-shift explanation the proposal raised as a hypothesis in Section 1, now with the numbers behind it. It is still a correlational account of the training data, not a controlled intervention; Step 3 is where an intervention would go.

## 4. More search helps, and does not change what goes wrong

The three Greedy pairings were replayed with the same openings, the same per-game seeds and therefore the same Greedy tie-breaks, changing only the network's simulations per move.

| simulations | gen60: score vs Greedy [95%], wipeouts | control120 | gen120 |
|---:|---|---|---|
| 50 (pilot) | 64.5% [57.7, 70.8], 70 | 64.0% [57.1, 70.3], 72 | 82.0% [76.1, 86.7], 35 |
| 256 | 76.5% [70.2, 81.8], 46 | 84.5% [78.8, 88.9], 31 | 93.0% [88.6, 95.8], 14 |
| 800 | 88.5% [83.3, 92.2], 23 | 95.5% [91.7, 97.6], 9 | 97.0% [93.6, 98.6], 6 |

The 256-simulation replays ran on the laptop (about 47 minutes per pairing); the 800-simulation replays ran on the CSSE cluster's CPU partition, one core per pairing, about two hours each.

Each step up in search budget removes a third to a half of the wipeouts that were left: 70, 46, 23 for gen60; 72, 31, 9 for control120; 35, 14, 6 for gen120. Sixteen times the pilot's budget still leaves gen60 losing more than one game in ten to Greedy. Every loss that remains, at either budget, is still a wipeout, still avoidable at the second-to-last move in every case, and, in all but two of the 29 games at 800 simulations where a safe last move existed, the search had visited a safe move. The searched value at gen60's last move stays above +0.9 at both budgets. gen120 at 800 simulations is the one place the search catches the loss: in four of its six wipeouts the searched value turned negative at the last move, but in none did it turn negative a move earlier, when every one of them was still avoidable. Search compensates for the value estimate in proportion to how much of the tree it can cover; it does not correct the estimate.

## 5. The ownership head's effect holds on a second seed

Step 1 found that at generation 120 the network trained with the ownership head (gen120) is wiped out half as often as the one trained without it (control120), with one training run per recipe. A second training run of the ownership recipe with a different random seed (`e2-ownership-seed2`, Run 7 in [experiments.md](../docs/experiments.md)) was played under the pilot protocol: 200 games against each of Greedy and Minimax-d4, 50 simulations, 4-ply openings, tournament seed 20261025 (report: [`results/step2/pilot-seed2.json`](results/step2/pilot-seed2.json)).

| checkpoint | score vs Greedy [95%] | wipeouts | score vs Minimax-d4 [95%] |
|---|---|---:|---|
| control120 (no ownership head) | 64.0% [57.1, 70.3] | 72 | 96.5% [93.0, 98.3] |
| gen120 (ownership head, seed 1) | 82.0% [76.1, 86.7] | 35 | 99.0% [96.4, 99.7] |
| seed2 (ownership head, seed 2) | 76.5% [70.2, 81.8] | 47 | 97.0% [93.6, 98.6] |

The second seed lands between the two: better than the control, below the first ownership seed. The direction holds, the size varies by seed, so "the ownership head halves the wipeouts" is one seed's number and "the ownership head cuts them by a third to a half" is what two seeds support. The per-pairing openings are derived from the tournament seed and the entrant names, so seed2 saw a different set of 200 openings than gen120 did; the comparison is between two independent samples, each with its own interval.

What the second seed loses looks exactly like Sections 1 and 2: 47 losses, all wipeouts, searched value +0.90 at the last move and still positive in 46 of the 47, avoidable at the second-to-last move in every case, with a safe move the runner-up in 32 of the 41 games where one existed at the last move. In one game the opening left the network with a single disc and Greedy moved first, so the network never moved; the tables that need an estimate leave that game out.

## 6. What this does and does not establish

Established, each by a different piece of evidence:

- The losses are early wipeouts and nothing else (Step 1, 3,000 games).
- The network's value estimate is near +1 in the positions it loses from, before and after search, and the won games get the same estimate (Section 1).
- An escape existed and was within the search's view in nearly every case (Section 2).
- Self-play produced such positions about once in a hundred games and labelled them as winning (Section 3).
- Raising the search budget reduces but does not remove the losses, and leaves their character unchanged (Section 4).
- The ownership head reduces the wipeouts by a third to a half on two training seeds, and what remains has the same character (Section 5).
- The value head has the sign wrong one ply before the wipeout, and the policy head gives the wiping reply a few percent; at fifty simulations the prior is what lets the loss through, and the value head is what makes seeing the wipeout necessary (section 7).
- Two other disc-hungry opponents produce the same wipeouts, with the same estimates behind them, and a two-ply disc count beats the generation-60 network outright (section 8).

Not established:

- That changing the training distribution would fix it. Section 3 is a description of the data, not an intervention. The ownership head's effect (Section 5) is the closest thing to one: it holds on two seeds of that recipe, but the control recipe still has one seed, and the head changes the learning target, not the opponents, so it says nothing about opponent diversity.

Sections 7 and 8 were added after the above was written. Section 7 settles which head lets the loss through, which the first version of this document left open; section 8 confirms that the loss is not specific to Greedy.

## 7. Which head lets the wipeout through

Section 2 left two suspects. The *value head* rates the position, and the one a move later with the network holding a single disc, as a near-certain win, so the search has no reason to steer away. The *policy head* assigns Greedy's wiping reply a prior, and if that prior is small enough, fifty simulations never visit the reply and never discover that the game ends there. The recorded games cannot separate the two, because they hold the root's visit counts and not the network's prior over the opponent's replies.

[`probe_heads.py`](../scripts/probe_heads.py) asks the network directly. For every wipeout in the pilot where a surviving move existed at the network's last move (169 positions across the three networks), it reads the raw value and the prior at that position, then plays the fatal move and reads the raw value and the prior from Greedy's side, where the exact value is +1 because a wiping reply exists. Then it runs the same 50-simulation search three ways on the same position: as recorded; with the prior flattened to uniform, so only the value head speaks; and with the value zeroed, so only the prior speaks (a finished position still scores exactly, so a search that reaches the wipeout sees the -1). Report: [`results/step2/head-probe.json`](results/step2/head-probe.json).

| network | positions | raw value at the fatal position | prior on the move played | prior on all surviving moves | one ply later, raw value for Greedy (exact: +1) | prior on Greedy's wiping replies, median | below 5% | a wiping reply had Greedy's top prior |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gen60 | 68 | +0.99 | 46% | 26% | -0.99 | 3.8% | 65% | 1% |
| control120 | 70 | +0.99 | 39% | 32% | -1.00 | 3.4% | 63% | 0% |
| gen120 (ownership head) | 31 | +0.98 | 39% | 37% | -0.96 | 5.8% | 29% | 0% |

Both heads are wrong, and in the same direction. The value head puts the fatal position at +0.99 and, from Greedy's side one ply later, at -1.00 where the truth is +1: it does not merely underrate the danger, it has the sign backwards. The policy head gives the wiping reply a few percent, and in no network is it ever Greedy's most-likely move. From the network's point of view, Greedy is about to play something reasonable in a position the network is winning.

| network | as recorded, 50 sims | value head only | policy head only | as recorded, 256 sims | value head only, 256 | policy head only, 256 |
|---|---:|---:|---:|---:|---:|---:|
| gen60 | 0% | **99%** | 3% | 51% | 100% | 47% |
| control120 | 0% | **100%** | 1% | 60% | 100% | 60% |
| gen120 | 0% | **100%** | 3% | 81% | 100% | 81% |

Share of the probed positions where the search chooses a surviving move. The as-recorded search reproduced the move in the record in every position, so the probe is looking at the same search the pilot ran.

**At fifty simulations the prior is what decides it.** Flatten the prior and the search survives 99% to 100% of these positions, with the same wrong value head: once the wiping reply gets an equal share of the opponent's visits, the search reaches the end of the game under the fatal move, scores it exactly, and prefers the move it was already giving a quarter to a third of its visits. Zero the value instead and the search still walks into the wipeout 97% to 99% of the time, because the prior alone steers the opponent's visits away from the reply. At 256 simulations the recorded search survives about half the positions, and the policy-only search does the same, which says that the extra budget helps only by spending enough visits at Greedy's node to reach a reply with a 3% prior.

So the chain is: the value head is confidently wrong about few-disc positions (sections 1 and 3), which is why the search needs to *see* the wipeout to avoid it, and the policy head is what stops it seeing one in time. Either head alone would be enough to lose by; the prior is the one that fifty simulations cannot work around.

**What this means for an intervention.** A training change that only sharpened the policy head's idea of what Greedy plays would fix the symptom at the opponent's node and leave the value at +0.99, and the policy head is trained only on positions where the network itself is to move, so a game against Greedy does not even produce a policy target at Greedy's node. The value target is the lever a mixed-opponent game actually pulls: a position where the network holds one disc, labelled with the loss that followed, trains the value head toward the right sign, and a value head that rates the fatal position below the safe one does not need the search to reach the end. That is the reason Step 3 uses external games for both targets rather than for the policy target alone, and the reason the ownership head, which changes the value signal and nothing about the prior, moved the number at all.

## 8. It is not Greedy. It is any opponent that goes after discs

Greedy is one opponent, and a weakness against one opponent could be a quirk of that opponent. Two variants were built to check (`src/reversi/agents/greedy.py`): **greedy-corner**, which takes a corner whenever one is available and otherwise flips the most discs, and **greedy-d2**, a two-ply search that counts nothing but discs, so it is Greedy that sees the opponent's greedy reply coming. Both still go after discs most of the time; neither is Greedy. The three checkpoints played both under the Step 1 protocol (200 games, 50 simulations, the same 4-ply openings), with Greedy and Random in the field so the pilot's pairings were replayed as a check: every Greedy pairing reproduced the pilot's score, wipeout count and wipeout lengths exactly, as the seed derivation promises. Report: [`results/step2/disc-hungry.json`](results/step2/disc-hungry.json), played on the cluster in 33 minutes.

| checkpoint | rating, Random = 0 | vs Greedy (wiped out) | vs greedy-corner (wiped out) | vs greedy-d2 (wiped out) |
|---|---|---:|---:|---:|
| gen60 | +364 | 64.5% (70) | 72.5% (55) | **56.0% (88)** |
| control120 | +405 | 64.0% (72) | 66.0% (68) | 63.0% (74) |
| gen120 (ownership head) | +514 | 82.0% (35) | 78.0% (43) | 80.5% (39) |
| greedy-corner | +292 | 78.2% | | 52.0% |
| greedy-d2 | +292 | 69.0% | 48.0% | |
| Greedy | +183 | | | |

Ratings are from this tournament's own fit and are not comparable to the pilot's.

**The pattern holds for all three, and the weakest version of Greedy is not the most dangerous one.** Every loss to greedy-corner and to greedy-d2 but one is a wipeout: 166 of 167 and 201 of 201 across the three networks, and no loss went the distance. The two variants are stronger players than Greedy (both beat it about three times in four, and both rate 110 points above it), and against the self-play networks they are at least as dangerous: greedy-d2 wipes out gen60 88 times in 200 games and wins the pairing, 56% to 44%. A two-ply disc count, with no idea of corners, mobility or anything else, beats the generation-60 network that beats depth-4 Minimax 96% of the time.

The wipeouts look exactly like Greedy's. From [`analyze_wipeouts.py --opponent greedy-d2`](../scripts/analyze_wipeouts.py): the searched value at the network's last move is +0.91 (gen60), +0.92 (control120) and +0.65 (gen120), still positive in 86%, 99% and 82% of the games; the raw network value is +0.99 or +1.00 in all three; a surviving move existed at the second-to-last move in 198 of 201 games; and where a safe move existed at the last move it was the runner-up in roughly half. Against greedy-corner the numbers are the same within a few points. The one place greedy-d2 differs is that its two-ply search sometimes sets up a wipeout the network cannot escape at the last move (7 of 74 for control120, 9 of 39 for gen120), which is why it is the stronger of the two.

**The ownership head's effect holds for the family.** gen120 is wiped out a third to a half less often than control120 by every one of the three opponents: 35 against 72, 43 against 68, 39 against 74. Whatever the ownership target teaches, it is not specific to Greedy's exact move choice.

For Step 3 this settles two things. The weakness is a property of the network, triggered by any opponent that flips as many discs as it can, so a held-out roster that includes a disc-hungry opponent the network did not train against is a fair test of transfer within that family. And greedy-corner and greedy-d2 agree with Greedy's move in 88% and 74% of mid-game positions (the similarity check in [held-out-opponents.md](held-out-opponents.md)), so they are the training opponent's own family and are reported as such, never as held out.

## 7. For the report

The one-paragraph version: an AlphaZero-style Othello agent trained only by self-play loses a third of its games to a one-ply greedy opponent, every one of them by being wiped out early, while beating a depth-4 minimax search 96% of the time. The agent values a position where it holds one disc at +0.99 because, in a million self-play positions, holding one or two discs after the opening happened about once per hundred games and the side holding them won three quarters of the time. The rule it learned is correct for the games it saw; the opponent it was never shown is the one that makes the rule fatal.
