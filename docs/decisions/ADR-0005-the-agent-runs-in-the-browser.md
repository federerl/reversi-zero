# ADR-0005: The agent runs in the browser, not on a server

**Status:** accepted
**Applies to:** `web/`, `nn/onnx.py`, `web/fixtures.py`, `web/manifest.py`
**Supersedes:** nothing — `api/` is kept, see "What we did not do"
**Contracts:** C1 (perspective), C2 (value sign), C3 (pass and terminal), C5 (illegal moves)

## The problem

The project needed a playable demo that could be sent to somebody. The obvious
build — a FastAPI process holding the network, answering `/api/ai-move` — was
already written and tested, and it does not survive contact with real users.

Three reasons, in increasing order of how badly they hurt.

**A move costs a core-second.** A search at the top level is 800 simulations.
Measured through the API, the Python search runs about 255 simulations per
second, so one move is roughly three seconds of one CPU core. The service allows
two concurrent searches and returns `429` past that. Three friends playing at
once is already over the limit.

**Free hosting sleeps.** As of 2026, Render's free web services spin down after
15 minutes idle and take 30–50 seconds to wake; Railway has no permanent free
tier; Fly.io no longer offers one to new accounts. A link you send someone is,
by definition, opened after an idle period. Their first experience would be a
blank page for most of a minute, then three-second moves.

**The interesting part is invisible.** The actual result of the project — that
the agent got measurably stronger, +547 Elo at generation 5 to +877 at
generation 60 — was a chart in a document. A visitor played one opponent at one
strength and had no way to feel any of it.

All three come from one decision: the agent runs on *our* machine, so the number
of people who can use it is bounded by what we pay for, and what they can be
shown is bounded by what we can afford to compute.

## The decision

**Ship the network to each visitor and run everything in their browser.** The
rules, the tree search and the neural network all execute client-side. The site
becomes static files.

This was decided on a measurement, not a preference. The trained network is
458,696 parameters, and every layer in it is an ordinary convolution, batch
norm, ReLU, linear or tanh — nothing exotic to port. Exported to ONNX it is
1.76 MB, about the size of a photograph, and it agrees with PyTorch to
6×10⁻⁶ in the policy logits.

That is small enough to give away rather than rent a machine to hold.

## Why this is not obviously right

Two objections deserve answering, because both are correct as far as they go.

**"You now have three implementations of the rules."** Yes, and that is the real
cost of this decision. See below.

**"WebAssembly is slower than native."** Also yes, and the first measurement
looked far worse than expected: 6.9 ms per position in the browser against
0.30 ms natively, a factor of twenty.

That comparison was wrong, and the mistake is worth recording because it nearly
sank the design. Native ONNX Runtime uses every core by default; the browser was
using one. **Per core the gap is about 2×**, which is the ordinary WebAssembly
penalty and exactly what was predicted. Single-threaded browser throughput is
about 160 positions per second.

A second measurement then ruled out the obvious remedy. Throughput is *flat*
from batch 1 to batch 32 — 146 to 170 positions per second — so the cost is
arithmetic rather than the overhead of crossing into WebAssembly. Batching
search leaves, which is what makes self-play fast on the GPU, gains nothing at
all here. Threads were the only lever, and they roughly halve everything:

| Simulations | 1 thread | 4 threads |
|---|---:|---:|
| 16 | 127 ms | 55 ms |
| 64 | 450 ms | 221 ms |
| 256 | 1,889 ms | 898 ms |
| 800 | 5,503 ms | 2,789 ms |

## The consequence that mattered most

**A simulation count is a fixed amount of work, not a fixed amount of waiting**,
and those differ by more than tenfold between a laptop and a phone. At 2.8
seconds the top level was past the point where a move reads as a hang rather
than as thinking.

So the deep levels take a deadline instead: `Strong` searches for up to 1.2
seconds and `Max` for up to 2.0, spending however many simulations fit. On the
machine measured above, `Max` reaches about 560 of its 800. The interface
reports the number actually reached and labels the level by its time, because
claiming 800 on a device that manages 300 would be the interface overstating how
hard the agent thought.

This was not an improvisation — the original plan specified "800 simulations
**or** a 1.5 s budget, whichever comes first". The measurement is what turned
that from a footnote into the design.

## The cost: a third implementation of the rules

The strongest thing about this repository is that the rules were *proved* rather
than assumed — two independent implementations, compared at every move over
20,000 random games nightly, zero mismatches, then frozen. Every strength number since
rests on that.

A TypeScript port is a third implementation, and a subtly wrong one fails in the
worst available way: **it does not crash.** The browser plays a slightly
different game from the one the agent learned, the agent looks weaker than it
is, and nothing reports a problem.

So the port is held to the same standard, with one rule doing the work:

> **The expectations are generated from the frozen engine, never written by
> hand.** A hand-written test encodes what somebody believed the rules were. A
> generated one encodes what the engine actually does — which is what the agent
> was trained against.

`reversi export-fixtures` emits five files: 1,000 positions with their legal
moves and the exact set of discs each move flips; the input encoding as bitmasks
so the comparison is per-square rather than per-total; the exported network's own
outputs; search visit counts from a deterministic evaluator with no exploration
noise; and 20 whole games replayed move by move. CI regenerates them and fails
on any difference.

### It caught a bug immediately

The first run failed, on the hazard this decision most obviously invites.
Building the board mask computed `(1 << 32) - 1` for a full 32-bit half — but
JavaScript takes shift counts **modulo 32**, so `1 << 32` is `1`, not 2³², and
the expression yields `0` rather than all ones.

The engine kept working. It stopped believing in the bottom four rows of the
board and offered one legal opening move instead of four. In a game that would
have read as a weak agent, not a broken port.

### The search agrees exactly, which was not expected

All 40 search positions reproduce the Python visit counts exactly. The two
runtimes use different implementations of `exp`, each entitled to differ by a
unit in the last place, and one flipped comparison early in a tree changes every
count after it.

Exact agreement is achievable because the port rounds the evaluator's outputs to
float32 with `Math.fround` at the points where Python stores them in a float32
array. Without that the priors diverge in their last digits from the first node
onward. The test asserts 90% rather than 100%, because a different browser
engine may land on the other side of one comparison; below that is a regression
rather than rounding.

## What we did not do

**Delete the FastAPI server.** It stays behind the same `Engine` interface as
the browser one, for about forty lines. It is a real fallback on a device where
WebAssembly is blocked, it is how the browser's answers get compared against the
reference during development, and it is where two-player games would attach.
Removing it would save nothing and close all three doors.

**Choose GitHub Pages.** It deploys from the repository already and would have
been simpler. It cannot set HTTP response headers, and this site needs
`Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` — without them
the runtime falls back to one thread and every move takes twice as long. That
decided it. Cloudflare Workers being on the same platform as Cloudflare Pages is
a second, softer reason: multiplayer later becomes an addition rather than a
migration.

**Use WebGPU.** Its build is twice the download, and for a network this small
the per-operation dispatch overhead usually costs more than the arithmetic it
saves. Revisit only with a measurement.

## Revisit if

* A search on a mid-range phone exceeds about two seconds at `Casual` — then
  leaf-batching is still useless, but a smaller network for mobile is not.
* Two-player games are wanted. That genuinely needs a server, and this decision
  does not prevent it.
* The `network` fixture starts failing after a runtime upgrade, which would mean
  ONNX Runtime changed an answer rather than that we did.
