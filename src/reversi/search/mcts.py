"""PUCT search: spending a few hundred network calls to find a better move.

One simulation, four steps:

1. **Descend.** From the root, repeatedly pick the most promising move until you
   reach a position the tree has not seen before.
2. **Ask the network** what it thinks of that new position -- unless the game is
   over there, in which case we know the real answer and use it.
3. **Carry the answer back up**, flipping its sign at every step, so each move
   along the way records how well it turned out.
4. Repeat.

At the end, the move that got visited most is the search's answer. Not the move
with the best average score -- most-visited. Those differ, and most-visited is
the right one: a move can score well on two lucky simulations, but it can only
accumulate visits by continuing to look good as the search examines it harder.
Visit counts are the estimate that gets more reliable the longer you search.

**Which move to look at next.** The rule balances three things:

    score(move) = Q(move) + c_puct * P(move) * sqrt(total visits) / (1 + visits(move))

``Q`` is how well the move has actually worked out so far, ``P`` is how much the
network liked it before we tried anything, and the fraction shrinks as a move
gets visited. Early on the network's opinion dominates; as visits accumulate,
measured results take over. That is the whole reason a few hundred simulations
beat a few hundred random rollouts -- the network points the search at the
handful of moves worth examining.

**The sign convention, which is the thing to get right.** Every value in this
file -- the network's, a terminal result's, every ``Q`` -- is from the point of
view of the player to move *at that node*. So a value coming back up the tree
must be negated once per level: a position that is winning for me is losing for
my opponent, who is the one choosing the move that led here. Contract C2, and
one line of code. Getting it wrong does not crash and does not show up in the
training loss; the search simply prefers moves that lose, and the agent trains
happily toward being bad. That is why it gets four dedicated tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray

from reversi.errors import SearchError
from reversi.game import rules, scoring
from reversi.game.state import State
from reversi.search.config import SearchConfig
from reversi.search.evaluator import Evaluator
from reversi.search.node import Node, priors_from_logits
from reversi.types import Action, policy_size

__all__ = ["MCTS", "SearchResult", "select_move"]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """What one search found. Aligned lists: entry ``i`` describes ``actions[i]``."""

    actions: tuple[Action, ...]
    visits: tuple[int, ...]
    q_values: tuple[float, ...]
    root_value: float
    """The network's own opinion of the root, before any searching. Worth keeping
    separately: comparing it against the searched result is how you tell whether
    the search is adding anything."""
    board_size: int

    @property
    def total_visits(self) -> int:
        return sum(self.visits)

    def best_action(self) -> Action:
        """The most-visited move. Ties go to the lower action index, so this is
        deterministic given the same tree."""
        best = max(range(len(self.actions)), key=lambda i: self.visits[i])
        return self.actions[best]

    def value(self) -> float:
        """The search's estimate for the player to move at the root, in [-1, +1]."""
        total = self.total_visits
        if total == 0:
            return self.root_value
        return sum(q * n for q, n in zip(self.q_values, self.visits, strict=True)) / total

    def policy_target(self) -> NDArray[np.float32]:
        """The training target: raw visit counts as a distribution over all actions.

        **Always the plain visit share, never temperature-adjusted** (contract
        C4). The temperature in this module affects which move gets *played*, and
        that is a completely separate thing -- see ``select_move``. Mixing the two
        up is the most common way to reimplement AlphaZero incorrectly: the
        network ends up trained on a sharpened or flattened version of what the
        search actually found, so it learns something the search never said.

        Full width (one entry per square, plus PASS), zero on illegal moves.
        """
        target = np.zeros(policy_size(self.board_size), dtype=np.float32)
        total = self.total_visits
        if total == 0:
            return target
        for action, visits in zip(self.actions, self.visits, strict=True):
            target[action] = visits / total
        return target

    def visit_counts(self) -> NDArray[np.int64]:
        """Full-width visit counts, for the web UI's heatmap and for debugging."""
        counts = np.zeros(policy_size(self.board_size), dtype=np.int64)
        for action, visits in zip(self.actions, self.visits, strict=True):
            counts[action] = visits
        return counts


def select_move(
    result: SearchResult,
    *,
    temperature: float,
    rng: np.random.Generator | None = None,
) -> Action:
    """Choose the move to actually play. Separate from the training target on purpose.

    ``temperature`` of 0 means play the most-visited move. Above 0, sample in
    proportion to ``visits ** (1 / temperature)`` -- so 1.0 samples in proportion
    to the visits themselves, and smaller values sharpen toward the best move.

    Why sample at all: without it, a network playing itself produces nearly the
    same game every time, and a training set of one game repeated thousands of
    times teaches nothing. Randomising the opening plies is what makes self-play
    cover different positions. It is turned off after ``temp_moves`` plies,
    because randomness in the middlegame just adds noise to the value targets.
    """
    if temperature <= 0.0:
        return result.best_action()
    if rng is None:
        msg = "sampling a move needs an rng; pass one or use temperature=0"
        raise SearchError(msg)

    counts = np.asarray(result.visits, dtype=np.float64)
    if counts.sum() <= 0:
        msg = "cannot sample a move from a search with no visits"
        raise SearchError(msg)

    weights = counts ** (1.0 / temperature)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        # A very small temperature overflows the exponentiation. That is just an
        # extreme way of saying "take the best move", so do that.
        return result.best_action()

    index = int(rng.choice(len(result.actions), p=weights / total))
    return result.actions[index]


class MCTS:
    """The search itself. One instance per agent; ``run`` builds a fresh tree per move.

    Trees are not reused between moves yet. Reuse is a measured optimisation for
    later (it also needs care: the promoted root must get *fresh* exploration
    noise, not noise layered on last move's noise).
    """

    __slots__ = ("config", "evaluator")

    def __init__(self, evaluator: Evaluator, config: SearchConfig | None = None) -> None:
        self.evaluator = evaluator
        self.config = config if config is not None else SearchConfig()

    # -----------------------------------------------------------------

    def run(self, state: State, *, rng: np.random.Generator | None = None) -> SearchResult:
        """Search from ``state`` and report what was found.

        ``rng`` is required when the config asks for exploration noise, and
        unused otherwise. Requiring it rather than quietly skipping the noise
        means a self-play worker that forgot to pass one fails immediately,
        instead of generating a whole generation of games with no variety in
        them.
        """
        if rules.is_terminal(state):
            msg = (
                "cannot search a finished position: there is no move to choose. "
                f"Check is_terminal() before searching.\n{state}"
            )
            raise SearchError(msg)

        root = Node(state)
        self._expand(root)

        if self.config.uses_noise:
            if rng is None:
                msg = (
                    f"dirichlet_eps={self.config.dirichlet_eps} asks for exploration "
                    "noise, but no rng was given to draw it from. Pass one, or use "
                    "SearchConfig.for_evaluation() if this is not self-play "
                    "(contract C7)."
                )
                raise SearchError(msg)
            self._add_root_noise(root, rng)

        for _ in range(self.config.n_simulations):
            self._simulate(root)

        return SearchResult(
            actions=root.actions,
            visits=tuple(root.visits),
            q_values=tuple(root.q(i) for i in range(root.n_actions)),
            root_value=root.net_value,
            board_size=state.size,
        )

    # -----------------------------------------------------------------
    # One simulation
    # -----------------------------------------------------------------

    def _simulate(self, root: Node) -> None:
        node, path = self.descend(root)
        self._backup(path, self._leaf_value(node))

    # -----------------------------------------------------------------
    # The pieces of one simulation, exposed so that a batched scheduler can
    # drive many trees with the *same* logic rather than a second copy of it.
    #
    # `run` above is one scheduler: descend, evaluate, back up, repeat, one
    # position at a time. `selfplay.game_batch` is another: descend every tree,
    # evaluate all their leaves in one call, back them all up. The rules of the
    # search live here and are shared; only the scheduling differs. Two copies
    # would drift, and a drift between them would be invisible -- both would
    # still produce legal moves.
    # -----------------------------------------------------------------

    def descend(self, root: Node) -> tuple[Node, list[tuple[Node, int]]]:
        """Walk from the root to a position the tree has not opened up yet.

        Returns the leaf and the path taken, which is what backup needs. Creates
        child nodes along the way as they are first reached.
        """
        node = root
        path: list[tuple[Node, int]] = []

        while node.expanded:
            index = self._select(node)
            path.append((node, index))
            child = node.children[index]
            if child is None:
                child = Node(rules.apply(node.state, node.actions[index]))
                node.children[index] = child
            node = child

        return node, path

    def terminal_value(self, node: Node) -> float | None:
        """The exact value of a finished position, or None if the game goes on.

        A batched scheduler calls this first so that finished positions never
        take up a slot in the network batch -- we know their answer exactly, and
        a guess could only be worse *and* slower.
        """
        if not rules.is_terminal(node.state):
            return None
        if node.terminal_value is None:
            node.terminal_value = float(scoring.result(node.state))
        return node.terminal_value

    def open_node(
        self,
        node: Node,
        logits: NDArray[np.float32],
        value: float,
    ) -> float:
        """Attach the legal moves to a leaf, using an answer obtained elsewhere.

        Split out from ``_expand`` so a batched scheduler can hand back one row
        of a batch it already ran.
        """
        actions = rules.legal_actions(node.state)
        node.open_with(actions, priors_from_logits(logits, actions), value)

        if __debug__:
            # Contract C5, layer 2: an edge for every legal move and no others,
            # so the search cannot reach an illegal move even in principle.
            assert set(node.actions) == set(actions), "expanded node has illegal edges"
        return value

    def backup(self, path: Sequence[tuple[Node, int]], leaf_value: float) -> None:
        """Public name for the sign-flipping walk back up -- see ``_backup``."""
        self._backup(path, leaf_value)

    def _select(self, node: Node) -> int:
        """Pick the child to descend into: highest Q + exploration bonus."""
        config = self.config

        total = node.total_visits
        # The textbook formula uses sqrt(total), which is 0 before any child has
        # been visited -- making the exploration term 0 for every move, so the
        # first simulation ignores the network entirely and takes whichever move
        # happens to be numbered lowest. Flooring the count at 1 makes that first
        # choice follow the network's preference, which is the point of having
        # one. Behaviour is identical from the second simulation onward.
        sqrt_total = sqrt(total) if total > 0 else 1.0

        # A move nobody has tried has no measured value, so it borrows the
        # parent's -- reduced a little, and reduced more once a decent share of
        # the parent's prior has already been explored.
        explored_prior = sum(node.prior[i] for i in range(node.n_actions) if node.visits[i] > 0)
        unvisited_value = node.mean_value() - config.fpu_reduction * sqrt(explored_prior)

        best_index = 0
        best_score = -float("inf")
        for index in range(node.n_actions):
            visits = node.visits[index]
            q = node.value_sum[index] / visits if visits else unvisited_value
            u = config.c_puct * node.prior[index] * sqrt_total / (1 + visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _leaf_value(self, node: Node) -> float:
        """The value of a freshly reached position, from *its* mover's point of view."""
        exact = self.terminal_value(node)
        if exact is not None:
            return exact

        self._expand(node)
        return node.net_value

    def _expand(self, node: Node) -> None:
        """Attach the legal moves and ask the network what it thinks. Batch of one."""
        logits, values = self.evaluator.evaluate([node.state])
        self.open_node(node, logits[0], float(values[0]))

    @staticmethod
    def _backup(path: Sequence[tuple[Node, int]], leaf_value: float) -> None:
        """Record the result along the path that produced it -- contract C2.

        ``leaf_value`` is from the leaf's mover's point of view. Each step up the
        tree crosses one turn boundary, so the sign flips exactly once per step:
        good for me is bad for the player who chose the move that got here.

        Passing does not need a special case. A pass changes whose turn it is
        just like a placement does, so the alternation holds through it.
        """
        value = leaf_value
        for node, index in reversed(path):
            value = -value
            node.visits[index] += 1
            node.value_sum[index] += value

    def _add_root_noise(self, root: Node, rng: np.random.Generator) -> None:
        """Mix random noise into the root's priors -- contract C7.

        Without this, a self-play game is fully determined by the weights: the
        same network plays the same game every time, and the training set is one
        game copied thousands of times. The noise makes the search give moves the
        network currently dislikes an occasional real look, which is the only way
        the agent ever discovers that its current preferences are wrong.

        Drawn fresh at every move's root and never accumulated. Only at the root:
        deeper in the tree, noise would just corrupt the evaluation of lines the
        search is trying to read accurately.
        """
        eps = self.config.dirichlet_eps
        noise = rng.dirichlet([self.config.dirichlet_alpha] * root.n_actions)
        root.prior = [
            (1.0 - eps) * prior + eps * float(sample)
            for prior, sample in zip(root.prior, noise, strict=True)
        ]
