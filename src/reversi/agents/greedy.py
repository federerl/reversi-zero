"""Takes whichever move flips the most discs right now.

A step up from random, and a useful one, because it is the obvious wrong strategy
in Reversi -- and beating it is genuinely informative.

**Why maximising flips is a bad idea.** Discs flip back. A move that turns over
eight of your opponent's pieces in the opening leaves you with a large, exposed
group and hands them the moves that flip it all back. Strong Reversi play does
close to the opposite: keep your disc count *low* in the early game, keep your
options open, and take the squares that cannot be flipped -- corners, and the
edges that lead to them. Greedy will happily give away a corner to gain three
discs in the middle.

So an agent that beats Greedy has learned something real: that the disc count on
move twenty does not predict the disc count on move sixty. That is not obvious
from the rules, and nothing in the training signal states it. It has to be
discovered from games.

**The variants further down exist for one question.** The trained network
loses to Greedy by being wiped out early (see ``research/``), and the question
is whether that is a weakness against Greedy in particular or against any
opponent that goes after discs. ``GreedyAgent(corner_first=True)`` is Greedy
that will take a corner when one is on offer; ``MaterialAgent`` is Greedy that
looks a few moves ahead, still counting nothing but discs; ``NoisyGreedyAgent``
is Greedy that sometimes plays at random. Each flips as many discs as it can
most of the time, and none is quite Greedy. They are not meant to be good.
"""

from __future__ import annotations

import numpy as np

from reversi.game import rules
from reversi.game.bitboard import popcount
from reversi.game.state import State
from reversi.types import Action, pass_action

__all__ = ["GreedyAgent", "MaterialAgent", "NoisyGreedyAgent", "corners"]


def corners(board_size: int) -> tuple[Action, ...]:
    """The four corner squares, as action indices."""
    last = board_size - 1
    return (0, last, last * board_size, last * board_size + last)


def _pick(choices: list[Action], rng: np.random.Generator) -> Action:
    """Break a tie uniformly at random.

    Ties are common and must not be broken by board order: always taking the
    lowest-numbered square would make an agent prefer the top-left corner of the
    board, which is a positional bias rather than the strategy it claims to
    follow, and it would quietly flatter any opponent that learns to exploit it.
    """
    return choices[int(rng.integers(0, len(choices)))]


class GreedyAgent:
    """Maximises discs flipped this move; ties broken uniformly at random.

    With ``corner_first``, a corner is taken whenever one is available, and
    flips decide only among the rest. That is the one piece of Othello knowledge
    everybody has, grafted onto an agent that otherwise has none.
    """

    def __init__(self, name: str | None = None, *, corner_first: bool = False) -> None:
        self.corner_first = corner_first
        if name is None:
            name = "greedy-corner" if corner_first else "greedy"
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def choices(self, state: State) -> list[Action]:
        """Every move this agent might play here: the tied best ones.

        Exposed so an analysis can know the whole tie set rather than sampling
        it, and so a test can check the tie-breaking is doing what it claims.
        """
        actions = rules.legal_actions(state)
        if not actions:
            msg = f"asked for a move in a finished position:\n{state}"
            raise ValueError(msg)
        if actions == [pass_action(state.size)]:
            return actions

        if self.corner_first:
            available = [a for a in actions if a in corners(state.size)]
            if available:
                return available

        gains = [popcount(rules.flips(state, action)) for action in actions]
        best = max(gains)
        return [action for action, gain in zip(actions, gains, strict=True) if gain == best]

    def select(self, state: State, rng: np.random.Generator) -> Action:
        return _pick(self.choices(state), rng)


class MaterialAgent:
    """Greedy that looks ``depth`` plies ahead, still counting only discs.

    A full-width minimax search whose only idea of a good position is "more of
    my discs than yours". At depth 1 it orders moves exactly as Greedy does
    (a move that flips ``f`` discs changes the difference by ``2f + 1``), so
    ``MaterialAgent(1)`` is Greedy with extra steps; at depth 2 it is Greedy
    that sees the opponent's greedy reply coming. Root ties are broken at
    random, as Greedy's are, for the same reason.
    """

    def __init__(self, depth: int = 2, name: str | None = None) -> None:
        if depth < 1:
            msg = f"depth must be at least 1, got {depth}"
            raise ValueError(msg)
        self.depth = depth
        self._name = name if name is not None else f"greedy-d{depth}"

    @property
    def name(self) -> str:
        return self._name

    def choices(self, state: State) -> list[Action]:
        """The moves tied for the best disc difference ``depth`` plies out."""
        actions = rules.legal_actions(state)
        if not actions:
            msg = f"asked for a move in a finished position:\n{state}"
            raise ValueError(msg)
        if len(actions) == 1:
            return actions
        scores = [-self._search(rules.apply(state, a), self.depth - 1) for a in actions]
        best = max(scores)
        return [a for a, s in zip(actions, scores, strict=True) if s == best]

    def select(self, state: State, rng: np.random.Generator) -> Action:
        return _pick(self.choices(state), rng)

    def _search(self, state: State, depth: int) -> int:
        """Disc difference for the player to move, ``depth`` plies further on."""
        if depth <= 0 or rules.is_terminal(state):
            return popcount(state.mine) - popcount(state.theirs)
        return max(
            -self._search(rules.apply(state, a), depth - 1) for a in rules.legal_actions(state)
        )


class NoisyGreedyAgent:
    """Greedy that plays a uniformly random legal move a fraction of the time.

    The random moves are the point: the question this helps answer is whether
    an opponent that *usually* goes after discs is as dangerous to the network
    as one that always does.
    """

    def __init__(self, epsilon: float = 0.25, name: str | None = None) -> None:
        if not 0.0 <= epsilon <= 1.0:
            msg = f"epsilon must be between 0 and 1, got {epsilon}"
            raise ValueError(msg)
        self.epsilon = epsilon
        self._greedy = GreedyAgent()
        self._name = name if name is not None else "noisy-greedy"

    @property
    def name(self) -> str:
        return self._name

    def select(self, state: State, rng: np.random.Generator) -> Action:
        greedy = self._greedy.choices(state)
        if rng.random() < self.epsilon:
            return _pick(rules.legal_actions(state), rng)
        return _pick(greedy, rng)
