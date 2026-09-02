# ADR-0003: Eight symmetries, including the four that cannot happen

**Status:** accepted, day 3
**Applies to:** `game/symmetry.py`, `data/replay.py`, `train/trainer.py`
**Contracts:** C6 (the eight symmetries)

## The problem

A Reversi board rotated 90° is the same position wearing a different hat. The
rules do not care which way up the board is, so a network that has learned about
one arrangement has, in principle, learned about all eight.

That is worth a great deal when training data is the scarce resource. 36,000
games produced two million positions; treating each as eight is the cheapest
multiplier available, and it costs nothing but a permutation.

The question is which eight, and whether all of them are legitimate.

## What is being claimed

The dihedral group D4 — four rotations and their four mirrors — acting on the 64
squares. For a symmetry `g`:

* **State:** permute the bits of both bitboards. `to_move` is unchanged.
* **Policy** (length 65): `out[perm[i]] = pi[i]` for `i < 64`, and
  **`out[64] = pi[64]`** — PASS is invariant under every symmetry, because PASS
  is not a square and no rotation moves it.
* **Value:** unchanged. Rotating a board does not change who is winning.

This holds because every rule of Othello commutes with D4: the flip directions
map onto each other, legality is preserved, termination is preserved, and disc
counts are untouched. So `π(g·s) = g·π(s)` and `v(g·s) = v(s)` for all eight.

## The subtlety worth writing down

**Only four of the eight stabilise the standard opening position.**

The four starting discs sit on a 2×2 diagonal arrangement. Rotating by 90°
produces the *other* diagonal — a position that is perfectly legal Othello and
cannot be reached from the standard start. So half of the symmetries map real
positions onto positions the agent will never actually face.

That means augmentation is not free in the way it first looks. It shifts the
training distribution slightly off the distribution the agent plays in: the
network spends some of its capacity on a mirror world.

## The decision

**Use all eight, and accept the shift.**

The correctness argument does not depend on reachability. `π(g·s) = g·π(s)` is
true because the rules commute with `g`, not because `g·s` is reachable. The
network is learning a function of a board, and the function genuinely has this
symmetry everywhere on its domain — including on positions no game will reach.

Using only the four that stabilise the opening would halve the multiplier to buy
a distributional purity that has no measured value. And there is a plausible
argument the other way: positions off the reachable manifold are still positions
where the same tactics apply, so learning them may generalise rather than waste
capacity.

**This is a judgement call, not a proof.** It is written down because a reader
should know it was made deliberately rather than overlooked, and because if the
agent ever behaves oddly in a way that smells distributional, this is one of the
first things to test — restrict to four and compare.

## Where it is applied

**At sampling time in the trainer, not at write time.** A random `g` is drawn per
sample, per epoch.

Two reasons. Eight times less disk, which matters when a generation is
50–150 MB of shards. And better epoch-to-epoch variety: the same stored position
appears under a different symmetry each time it is drawn, rather than the same
eight copies cycling forever.

It also keeps a change to the symmetry code from invalidating data already on
disk — the shards hold positions, not orientations.

## What could go wrong, and what catches it

**A permutation that is not its own inverse's inverse.** Tested: `g ∘ g⁻¹ == id`
for all eight.

**PASS being rotated.** Index 64 is not a square. If a permutation ever touched
it, the policy target would put PASS's probability on a random board square and
the network would learn to pass in positions where passing is illegal. Explicitly
tested.

**The encoder and the permutation disagreeing.** `encode(g·s) == g·encode(s)` is
tested directly, over ≥1000 Hypothesis examples, along with
`apply(g·s, g·a) == g·apply(s, a)` and `legal(g·s) == g·legal(s)`.

All five properties run against all eight operations. This is `-m property` in
the test suite.

## Revisit if

* A distributional problem is suspected. The experiment is cheap: restrict to the
  four opening-stabilising symmetries, retrain, compare ratings.
* The starting position becomes configurable. The stabiliser subgroup depends on
  it, and the "only four" observation above would need recomputing.
