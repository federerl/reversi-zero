# ADR-0006: The `Game` interface ships with the second game, not before it

**Status:** accepted
**Applies to:** `src/reversi/`, the 1.0 scope in `docs/roadmap-1.0.md`
**Supersedes:** workstream C of the 1.0 plan, which scheduled it inside 1.0
**Contracts:** C1, C2, C6 stay stated in Reversi terms until this lands

## The problem

The 1.0 plan reserved about eleven hours for a `Game` protocol: a
standard-library-and-numpy interface naming what the pipeline needs from a board
game — identity and shape, rules, encoding, storage, symmetry, and an optional
ownership target — with `ReversiGame` delegating to the existing modules and the
search, self-play, data and training layers each taking a game in turn. A 3×3
three-in-a-row game under `tests/support/` was to exercise the seam, so the
abstraction was tested by something other than the one implementation it was
extracted from.

The plan also put it on the "never cut" list, alongside the SLURM scripts, the
cross-generation tournament and the hub. It is the only item on that list whose
entire payoff falls outside 1.0.

Gomoku is 1.1. The interface exists so that Gomoku costs a directory rather than
a fork of the training loop. Nothing in 1.0 plays anything but Reversi, so
landing it now buys 1.0 no capability, no measurement and no user-visible change
— it buys 1.1 a cheaper start, four weeks early.

Against that, it is the single largest remaining item, and it touches the
checkpoint format, the shard format and the training loop: the three places where
a mistake is expensive to discover. It has to be done as four separate changes,
each green on all four gates, each proving Reversi's behaviour is byte-identical
— a shard's sha256 unchanged on a fixed seed, batched self-play still producing
the same games, the released `gen60.pt` still opening. That is careful work, and
careful work done against a deadline for a benefit that arrives later is how the
formats end up quietly wrong.

## The decision

**Move the `Game` interface, steps G1–G4 and the seam test, into 1.1, to be done
as the first work of that release, before any Gomoku-specific code.**

The remaining 1.0 scope is unchanged. The three other never-cut items that were
still open — move history and replay, manifest version 2, and the measurement of
difficulty spread — stay in 1.0.

This is recorded as a decision rather than left as an omission because the plan
explicitly forbade cutting it. A "never cut" item that quietly does not appear is
worse than either doing it or dropping it: a reader of the repository cannot tell
which happened, and neither can its author six months later.

## Why 1.1 is the right home, and not just the convenient one

**An interface extracted from one implementation is a guess.** The plan knew
this, which is why it specified a second game to exercise the seam. But
three-in-a-row on a 3×3 board is a test fixture, not a game anybody wants: it has
no pass move, no ownership, a nine-action policy and a two-plane encoding. It
would confirm that the interface is not *Reversi-shaped* while saying nothing
about whether it is *Gomoku-shaped*, and Gomoku is the game that has to fit.
Gomoku brings a 15×15 board, no captures, no passing, a policy 25 times larger,
and a terminal test that is a line scan rather than a disc count. Designing
against the real second game is strictly better information.

**The storage limitation is real and would have to be revisited anyway.** The
planned protocol keeps shards as "two bitboards and a side to move", which the
plan itself flagged as a stated 1.0 limitation. Two bitboards describe Reversi
because every square is empty, black or white. Gomoku fits that too. Chess does
not, and Chinese chess does not. So the format question the interface was going
to settle is not actually settled by it — it is deferred either way, and
deferring it next to the game that forces it is better than deferring it twice.

**Nothing in 1.0 is blocked.** The web side already has the structure that
matters for a second game: `src/hub/`, `src/shared/`, `src/games/reversi/`, three
Vite entry points, and a registry the hub reads. That refactor was structural for
1.1 and it shipped. The Python side has no equivalent dependency — Gomoku's
training does not exist yet, so there is nothing for it to be incompatible with.

## What this costs

**Gomoku starts with a refactor rather than with a game.** The first week of 1.1
is the interface and the four migration steps, with no visible progress. That is
a real cost and it is the honest one to pay: the alternative was paying it now,
also with no visible progress, in a release that has three open items whose
payoff is immediate.

**The contracts stay Reversi-worded for another release.** `docs/architecture.md`
states C1, C2 and C6 in terms of discs, players and a pass action. The plan
intended to restate them game-agnostically alongside the protocol. Until then a
reader may reasonably think the contracts are about Reversi specifically, when
what they are actually about — whose point of view a number is from, how a value
sign crosses an edge, which actions are invariant under symmetry — applies to any
two-player perfect-information game. This ADR is where that gap is recorded.

**The arena, the agents and the difficulty code stay Reversi-typed.** The plan
already made this a stretch item (G5) and it stays one. In 1.1 they are typed
against `Game` only where Gomoku needs them to be.

## What would change this

A second game arriving sooner than 1.1 — a request, or an interesting checkpoint
worth training on another board — makes the interface urgent again, and it should
be done before that game's first line of training code rather than after.
