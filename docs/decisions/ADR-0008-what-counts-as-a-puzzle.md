# ADR-0008: A puzzle is a position where the exact answer and the tempting move disagree

**Status:** accepted
**Applies to:** `src/reversi/endgame/puzzles.py`, `web/src/games/reversi/engine/__fixtures__/puzzles.json`, `/puzzles/`
**Contracts:** C3 (a game is over when neither side can place; empty squares go to nobody)

## The problem

The endgame solver returns the exact final disc difference after every legal move.
That makes a whole class of positions *answerable*. It does not make any of them
worth asking.

Most positions near the end of an Othello game are won by every legal move or lost
by all of them. Both are useless to a learner: one has no wrong answer, the other
has no right one. A page built by taking solved positions at random would be a
page of sixty questions of which perhaps five teach anything, and a player would
not be able to tell which five.

So something has to decide which positions are worth asking, and that decision
determines what the feature teaches. It is not an implementation detail of the
mining loop.

The difficulty is that "worth asking" is a claim about a *person* — about what
somebody would be inclined to play — and this repository has no person to
consult. Everything else it publishes is measured from play. A selection rule
that encoded one author's taste about which endgames are instructive would be the
one hand-typed judgement in a project whose whole argument is that it does not
have any.

## The decision

**A position is kept when the side to move has a win, and a move that looks
attractive throws it away.**

Both halves are computed, neither is assigned:

* **What is true** comes from the solver: the exact final disc difference after
  every legal move, with perfect play from both sides. `best > 0` or the position
  is not kept.
* **What looks attractive** comes from two models of a player who has not
  calculated. Either one is enough:
  * **the trained network's policy at the position**, one forward pass and no
    search — the move a reasonable player is drawn to before thinking. A search
    would find the right answer and stop being a model of the mistake.
  * **the move that flips the most discs**, which is the beginner instinct an
    endgame punishes hardest: discs taken now are nearly worthless because they
    can be flipped straight back.

A position also needs at least two legal moves, because a forced move is not a
choice.

For the disc-grabbing test, **every** move that flips the maximum number of discs
must lose. `GreedyAgent` breaks ties uniformly at random, so "grabbing discs loses
here" is only a fact about the position when no tie-break saves it. Taking one of
the tied moves instead would also make a committed artifact depend on a random
draw, which it must not.

**Difficulty is derived from the same two sources**, never labelled: how narrow the
win is (how many legal moves keep it), how much of the network's probability rests
on losing moves, and whether grabbing discs loses. An integer on 0–100, so a
shipped ordering cannot hinge on a float comparison.

The rule uses the network as an *instrument for finding mistakes*, which is a
different job from the one it was trained for and does not require it to be good.
It requires it to be wrong in the way people are wrong, which is what a policy
learned from self-play is.

## Why not the alternatives

**Hand-picked studies.** The classical way to build a puzzle set, and the
strongest in the hands of a strong player. It fails the standard the rest of this
repository is held to: every generated artifact here — the engine fixtures, the
ratings, the difficulty configuration — is produced from measurement, precisely so
that nobody has to trust the author's judgement about Othello. Nobody here is a
strong enough player for their taste to be worth trusting, and the honest response
to that is to not rely on it rather than to hide it.

**Any winnable position.** Cheap, and it produces a set where most entries have no
wrong answer. A player who solves fifteen in a row by clicking anything learns
that the page is not worth their attention, which is a worse outcome than a
smaller set.

**The largest swing.** Rank by `best - worst` and take the top N. This selects for
positions with a catastrophic move available, which correlates with being
instructive but is not it: a position where the disastrous move is also obviously
disastrous teaches nothing, and one where a subtle move loses by 2 teaches a great
deal. It also has no notion of temptation at all, which is the entire point.

**Positions where the agent itself blundered.** Appealing, and it would give the
puzzles a nice provenance. But the shipped agent rarely blunders outright in an
ending — the mining run bears this out, with the network's own first choice losing
in only 13 of 60 kept positions — so this would yield very few puzzles for a great
deal of play. It also selects for the agent's weaknesses rather than a person's,
and those are not the same.

**Ask the network for a difficulty score directly.** It has no such output, and
adding one would mean training a head against labels nobody has.

## What this does not do

**It does not claim the kept positions are the best sixty that exist.** They are
sixty that satisfy a stated rule, found by playing games until each stage's quota
was filled. A different seed finds a different sixty, all of which satisfy the
same rule. The rule is the claim; the particular set is a sample.

**It does not make the network an authority on difficulty.** Its policy is used as
a proxy for temptation, and the derived score says so by weighting it below the
one term that is exact — how narrow the win is.

**It selects on the first move, while the page asks for the whole ending.** The
rule here decides which *positions* are worth setting, and it reasons about the
move played from them -- that is what the network's policy and the disc-grabbing
test both speak to. The page then asks the player to win the ending, which is a
larger task than the rule measures. Difficulty is therefore a property of the
position as set, not a prediction of how hard the rest will be to convert, and
the ordering inside a stage should be read that way.

**It does not change how the agent plays.** The rule and the solver are analysis;
the ladder, the difficulty calibration and the published ratings are untouched.

## Consequences

The curriculum's character follows from the rule rather than from anyone's
intention, and is therefore worth reporting rather than assuming. In the shipped
set, 55 of 60 positions are ones where grabbing the most discs loses and 13 are
ones where the network's first choice loses. The disc-grabbing trap carries most
of the selection, because the shipped network is good enough at an ending that its
top move rarely throws the game away.

That is a fact about this network, not about the rule. A weaker network would
shift the balance the other way, and the file records which trap each position
sets so the interface can say which one it is.
