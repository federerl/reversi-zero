# Architecture

How the pieces fit together, which direction the dependencies run, and the seven
contracts that everything else assumes.

The last of those is the important part. Most of what can go wrong in a system
like this **fails silently** — no crash, no error, a loss curve that falls just
as nicely, and an agent that has quietly learned the wrong game. The contracts
are the specific places that happens, written down so they can be tested rather
than remembered.

---

## The layers

```
  game/          the rules. Pure Python, no torch, no numpy outside symmetry.py
    ↑
  search/        PUCT tree search. numpy only; the network sits behind a Protocol
    ↑
  nn/            the network, the feature encoding, export
    ↑
  agents/        Random, Greedy, Minimax, the trained agent, Edax
    ↑
  ┌──────────────┴───────────────┐
  arena/  difficulty/            selfplay/  data/  train/  ckpt/
  measuring strength             producing checkpoints
                                          │
  api/  web/   ◄──── an exported checkpoint, and nothing else
```

### The dependency rules

| Layer | May import | Must **not** import |
|---|---|---|
| `game/` | stdlib (numpy only in `symmetry.py`) | torch, search, nn |
| `search/` | `game`, numpy, the `Evaluator` protocol | torch directly |
| `nn/` | torch, numpy, `game` | search, selfplay |
| `agents/` | `game`, numpy; the trained agent also `search` + `nn` | selfplay, train |
| `selfplay/`, `train/`, `data/`, `ckpt/` | everything above | `api`, `web` |
| `arena/`, `difficulty/` | `game`, `agents`, `search`, `ckpt` | `train`, `selfplay`, `data.replay` |
| `api/` | `game`, `search`, `nn`, `agents`, `difficulty`, `ckpt` | **`train`, `selfplay`, `data`** |

These are **tested, not documented**. `tests/api/test_api.py` imports
`reversi.api.app` in a subprocess and asserts that no training module ended up in
`sys.modules`.

Three of these rules are load-bearing rather than tidy:

**`game/` must not import torch.** It is what lets the rules, the property tests
and the 20,000-game differential test run anywhere, including a CI runner with no
GPU and no torch build at all.

**`search/` must not import torch directly.** The network reaches it through an
`Evaluator` protocol, which is what allows a stub evaluator with a known answer.
Nearly every search test depends on that.

**`api/` must not reach `train/` or `data/`.** If a served game could reach the
replay buffer, the agent would be training on games played against its users —
data nobody chose, arriving through a public endpoint. Making the import
impossible is stronger than remembering not to write it.

### Where the browser fits

The web app re-implements `game/` and `search/` in TypeScript so the agent can run
on the visitor's machine. That is a third implementation of the rules, and it is
held to the same standard as the first two: every expectation it is tested against
is **generated** from the frozen Python engine. See ADR-0005 and `docs/web-app.md`.

---

## The seven contracts

Reproduced here in the form the plan states them. Each names the failure it
prevents, because in every case the failure is invisible without it.

### C1 — Perspective

Bit index `i = row·N + col`, row 0 at the top, column 0 at the left. The *same*
mapping in the engine, the feature encoder, the symmetry permutations, the API
JSON and the board on screen. `encode(state)` returns `[own, opp, legal_mask]`
for the player to move.

> **Test:** a position and its colour-swapped counterpart, with `to_move` also
> swapped, must encode to **identical** tensors.

*Why it fails silently:* a transposed board is still a board. The network learns
something, just not Othello, and no shape check would notice.

### C2 — Value sign in backup

The evaluator returns `v(leaf)` in the *leaf's* perspective. Walking leaf→root,
the value is negated **exactly once per edge**:

```python
value = leaf_value
for node, action in reversed(path):
    value = -value  # flip into the parent's perspective
    node.N[action] += 1
    node.W[action] += value
```

PASS also switches `to_move`, so no special case exists. `Q(a) = W(a)/N(a)` is
therefore always in the parent's perspective — exactly what the parent's argmax
needs.

> **Tests:** forced-win-in-1 → root `max Q ≥ 0.95`; forced-loss-in-1 → root
> `max Q ≤ −0.95`; a PASS-only position where the opponent then wins → root Q
> negative; a stub returning `v = 0` → all `Q == 0` exactly.

*Why it fails silently:* the search still returns legal moves. It simply prefers
losing ones, and the only symptom is an agent that appears not to learn. This is
the single most dangerous line in the project, and it has its own decision record
(ADR-0002).

### C3 — Pass, consecutive passes, terminal

```
placements = legal_placements(s)
if placements:                          legal_actions = placements ; terminal = False
elif legal_placements(opponent_of(s)):  legal_actions = [PASS]     ; terminal = False
else:                                   legal_actions = []         ; terminal = True
```

**"Two consecutive passes" is never a runtime state** — it collapses into
`terminal = True` at the first pass check. A full board is terminal by the same
rule, asserted rather than special-cased. `result(s) = sign(count(to_move) −
count(opponent)) ∈ {+1, 0, −1}`; empty squares go to nobody. `pass_count` is
tracked for metrics **only** and never consulted by the rules.

> **Test:** the Hypothesis invariant `terminal ⟺ legal_actions == []`.

### C4 — Visit counts → policy target

The stored target is **always** the raw visit distribution at τ=1:
`pi[a] = N(root,a) / ΣN` over legal `a`, 0 elsewhere, summing to 1 ± 1e-6.

The **move played** uses a *separate* temperature: `p_play[a] ∝ N(a)^(1/τ_play)`,
with `τ_play = 1.0` for the first `temp_moves` plies and → 0 afterwards.

> **These two must never share a variable.** Conflating them is the most common
> way to reimplement AlphaZero incorrectly: the network ends up trained on a
> sharpened or flattened version of what the search actually found, so it learns
> something the search never said.

When only one action is legal, the search is skipped, the move is played
immediately, and **no sample is stored**. `z` is back-filled at game end: for the
position at ply `t` with mover `m`, `z_t = result_for(m)`.

### C5 — Illegal actions blocked at three layers

1. **Network** — emits `N²+1` raw logits with **no internal masking**, which keeps
   the model a pure, exportable function. Training uses full-width cross-entropy
   against `π`, which is 0 on illegal actions, so the net learns to suppress them;
   **nothing ever relies on this.**
2. **Search** — edges exist only for legal actions; priors are a softmax over the
   legal subset, so illegal actions are unreachable *by construction*. There is
   nothing to mask, so a masking bug cannot exist.
3. **API** — the server re-derives the legal moves and returns **422** with the
   legal set for an illegal client move; the agent's own move is asserted legal
   before serialisation.

> **Tests:** 1000 scripted API games → zero illegal moves; 10k simulations on
> adversarial positions → zero visits on illegal indices.

### C6 — The eight symmetries

`SYMMETRIES` is 8 objects with `perm`/`inv` index arrays over D4.

* **State:** permute the bits of both bitboards; `to_move` unchanged.
* **Policy** (length `N²+1`): `out[perm[i]] = pi[i]` for `i < N²`, and
  **`out[N²] = pi[N²]` — PASS is invariant under every symmetry**, because PASS is
  not a square and no rotation moves it.
* **Value:** unchanged.

Reversi's rules commute with all of D4, so `π(g·s) = g·π(s)` and `v(g·s) = v(s)`
hold for all 8 regardless of reachability. **Only 4 of the 8 stabilise the
standard opening**; the other 4 produce positions that are legal but not reachable
from the standard start. That shifts the training distribution slightly off-policy
and is **accepted** — see ADR-0003.

Augmentation is applied **at sampling time** in the trainer: a random `g` per
sample per epoch. Eight times less disk, and better epoch-to-epoch variety.

### C7 — Root noise, exactly when

Applied **iff** `dirichlet_eps > 0` **and** the node is the current search root
**and** the caller is self-play. At the root, after legal masking and
normalisation:

```
P'(a) = (1−ε)·P(a) + ε·η(a),   η ~ Dir(α) over legal actions only
```

ε = 0.25, α = 1.0 on 8×8 (rule of thumb α ≈ 10/mean_branching), α = 2.0 on 4×4.
**Re-sampled fresh at every move's root, never accumulated.**

**ε = 0 in:** the arena, calibration, the API, the browser, and every test unless
explicitly parameterised. `arena/tournament.py`, `difficulty/calibrate.py` and
`api/service.py` each **assert** `eps == 0` at construction, because an agent
playing a rated match with exploration noise on is quietly playing worse than it
can — and the resulting rating would describe something nobody meant to measure.

---

## Why the correctness argument rests on three tests

Coverage is a floor, not evidence. The claim that this system implements Othello
correctly rests on:

**The differential test.** `game/reference.py` is a deliberately slow
list-of-lists implementation written from the rules text, and `game/rules.py` is
the fast bitboard one written independently from the same text. **20,000 random
games nightly** — roughly 1.2 million positions — compared at *every move*: legal
set, flip set, terminal flag, final score. Zero mismatches. Every push runs a
400-game version of the same test.

The plan asked for 50,000. The number was set by measurement instead: paired
games run at about 22 per second, so 50,000 would take ~38 minutes and not fit
the nightly budget alongside the other slow tests. 20,000 does, and the reasoning
is recorded in the test file rather than the figure quietly changed.

**The exact 4×4 solve.** The differential test proves the two engines agree. Two
engines can agree and both be wrong. So 4×4 Othello is also solved exactly (3,306
positions) and checked against the known result — **white wins with perfect
play**. That test never inspects a flip or a legal-move list; it plays every
possible game to the end and asks who wins.

> The differential test proves they agree. The solver proves the thing they agree
> on is Reversi.

**The 4×4 learning gate.** A 4×4 agent trained from random weights must reach
≥90% against random and ≥65% against greedy within 10 minutes of CPU. This is the
end-to-end check that the *whole pipeline* — encoding, search, storage, training,
checkpointing — does something, and it is the gate every later result depends on.

---

## Decision records

| | |
|---|---|
| ADR-0001 | Board representation: two integers, and what that costs |
| ADR-0002 | Whose point of view a number is from (C1, C2) |
| ADR-0003 | Symmetry and reachability (C6) |
| ADR-0004 | Process architecture: independent workers, synchronous trainer |
| ADR-0005 | The agent runs in the browser, not on a server |

## Related documents

| | |
|---|---|
| `docs/how-the-engine-works.md` | bitboards, passing, perspective, the rotations |
| `docs/training.md` | running, stopping, resuming, calibrating |
| `docs/experiments.md` | one entry per run: question, setup, outcome, decision |
| `docs/model_card.md` | compute, data provenance, measured strength, limitations |
| `docs/web-app.md` | the browser build and what "no server" means |
| `docs/configuration.md` | every setting and what breaks if you change it |
