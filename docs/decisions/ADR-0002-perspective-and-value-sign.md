# ADR-0002: Whose point of view a number is from

**Status:** accepted, day 4
**Applies to:** `nn/features.py`, `search/mcts.py`, `data/` (day 5), `arena/` (day 9), `api/` (day 13)
**Contracts:** C1 (perspective), C2 (value sign in backup)

## The problem

A Reversi position has two sides, so any number describing it needs an answer to
"good for whom?". `+0.7` means nothing on its own. It has to mean *+0.7 for
somebody*, and every part of the system has to agree on who that somebody is.

There are three plausible conventions:

1. **Always Black.** `+1` means Black is winning, wherever the number came from.
2. **Always the player to move** at the position being described.
3. **Always the root player** of whatever search is currently running.

They all work. What does not work is using more than one, and that is the easy
mistake to make: each one feels natural in a different file. Black's point of
view is natural when printing a board. The root player's is natural when the
search reports its answer. The player to move is natural inside the network.

## Why getting this wrong is dangerous

A sign error here has no symptom.

It does not crash — the number has the right type and stays in the right range.
It does not show up in the training loss — the network learns to predict whatever
it is shown, and if it is shown consistently inverted values it predicts those
accurately, so the loss falls just as nicely. It does not show up as an error
anywhere in the pipeline, because there is no downstream check that could know.

What actually happens is that the search prefers the moves it should avoid, the
agent trains toward being bad at Reversi, and the first evidence is an arena
result weeks later showing the "trained" agent losing to Random. At that point
every checkpoint, every measurement and every plot in the project is worthless,
and there is no way to tell from the artifacts which day the bug entered.

That risk profile — invisible, total, and late — is why this gets an ADR, an
assertion in the code, and four dedicated tests, rather than a comment.

## The decision

**Every value in this project is from the point of view of the player to move at
the position the value describes.** Option 2 above, with no exceptions.

Concretely:

| Where | What it means |
|---|---|
| `scoring.result(state)` | `+1` if the player to move at `state` won |
| `features.encode(state)` | plane 0 is the mover's discs, plane 1 the opponent's |
| the network's value head | `+1` if the player to move at the input position is winning |
| `Node.net_value`, `Node.q(i)` | from that node's mover's point of view |
| the replay buffer's `z` | from the point of view of whoever moved in that position |
| the API's `value` field | from the mover's view, and the payload says so explicitly |

Colours are **never** canonicalised in `State` — `black` always means black's
discs. The mover's-view conversion happens in exactly one place, the feature
encoder, so there is a single line to get right rather than five.

### The consequence that does the work

Because every value is relative to a node's own mover, and because every move
changes whose turn it is, **a value moving one step up the tree must have its
sign flipped exactly once**:

```python
value = leaf_value
for node, action in reversed(path):
    value = -value          # into the parent's point of view
    node.visits[action] += 1
    node.value_sum[action] += value
```

That is the whole of contract C2. `Q(a) = value_sum[a] / visits[a]` is then
automatically in the parent's point of view, which is exactly what the parent
needs in order to pick a move by taking a maximum.

**Passing needs no special case.** A pass changes whose turn it is just like a
placement does, so the alternation carries through it untouched. A "passes are
not real moves, skip the flip" shortcut is a real bug that this convention makes
unnecessary.

## What we gave up

**Debugging is slightly harder.** `+0.8` printed next to a board does not tell
you who is happy about it; you have to look at whose turn it is. Under the
always-Black convention you could read it directly.

We accepted that because the cost lands on a human reading a log occasionally,
while the benefit lands on every line of the search. Under always-Black, the
backup would have to flip conditionally on the node's colour, and every
comparison in the selection rule would need a colour-dependent sign. Conditional
sign logic that is *usually* right is precisely the failure mode described above.

The second reason is contract C1: with a mover's-view encoding, a position and
its colour-swapped twin are the *same input* to the network, so it learns Reversi
once instead of once per colour. That halving of the effective training data is a
real cost, and it only pays off if the value convention matches the encoding
convention.

## How it is checked

Four tests in `tests/unit/test_mcts.py`, each on a position found by searching
real games for the property rather than drawn by hand:

| Test | Position | Must hold |
|---|---|---|
| a winning move is `+1` | Black can end the game and win it | root `Q` for that move is `+1` |
| a lost position is `-1` | White's only legal move loses | root value is `-1` |
| the sign survives a pass | White must pass, then loses | root `Q` for PASS is negative |
| no opinion means exactly zero | stub returns `0` everywhere | every `Q` is exactly `0.0` |

The third is the one that would catch a pass-shaped special case; the fourth
would catch an accumulation bug that a sign test alone would miss.

`tests/unit/test_features.py` covers the encoding half: a position and its
colour-swapped twin, with the mover swapped too, must encode to byte-identical
arrays.

## Alternatives considered

**Always Black.** Rejected: pushes conditional sign logic into the search's
hottest path, and breaks the shared-encoding benefit above.

**Always the root player.** Rejected: the value has to change meaning as the tree
is walked, so the convention depends on context that is not visible from the node
you are holding. That is the hardest kind of convention to keep correct under
later refactoring.

**Storing a `perspective` field alongside every value.** Rejected as ceremony: it
makes the mistake detectable but does not prevent it, and it costs memory and a
branch in the hot loop. The single convention plus tests is cheaper and stronger.
