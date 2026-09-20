"""Finding positions worth being asked to solve, and arranging them into stages.

An exact answer is not yet a puzzle. Most positions near the end of an Othello
game are either won by every legal move or lost by all of them, and neither
teaches anything: one has no wrong answer and the other has no right one.

    A puzzle is a position where the exact answer and the tempting move
    disagree.

Two things are needed to say that, and this project already has both. The
**solver** says what is true -- the final disc difference after every legal move,
with perfect play from both sides. The **trained network's policy** says what
looks tempting, because it is a model of what a plausible player wants to play.
A position qualifies when the side to move has a win and a move that looks
attractive throws it away.

That rule is also a description of the complaint this feature exists to answer: a
game that was won with ten squares left, lost to one natural-looking move.

Two things count as "attractive", and they are different players:

* **the network's first choice**, which is the mistake a reasonable player makes;
* **taking the most discs**, which is the mistake a beginner makes, and the one
  the endgame punishes hardest. Discs flipped now are nearly worthless -- they
  can be flipped straight back -- and beginners weight them heavily anyway.

A position is kept when either of them loses a won game.

**Stages, not a pile.** The positions are arranged into five stages by how many
squares are still empty, because that is the honest difficulty axis and the
legible one: with four empty squares a player can count to the end in their head,
and with thirteen they cannot. It is also visible on the board, so somebody can
*see* why stage 4 is harder than stage 1 instead of being told. Within a stage
the order comes from a derived score (see ``difficulty_of``), never an assigned
one.

**Mining screens before it solves.** ``solve_root`` gives every move an exact
value, which is what a puzzle needs and roughly five times the work of finding
out whether a position is worth keeping at all. So each candidate is screened
with one or two plain ``solve_exact`` calls -- is this position even won, and does
the tempting move lose it -- and only a survivor pays for the full root solve.
Most candidates fail the first question.

**The output is not bit-reproducible, and that is why nothing diffs it.** Mining
plays real games with the trained network, so which positions turn up depends on
torch's floating-point arithmetic, which is not promised to be identical across
machines or library versions. Regenerating the puzzle file on another computer
may legitimately produce a different set of positions. What *can* be checked
anywhere, and is, is that every shipped puzzle's stored answers are the ones the
solver gives -- see ``tests/unit/test_endgame_puzzles.py``. Re-verifying is the
stronger check in any case: a hand-edited puzzle file is dangerous because it
would teach a wrong move, not because it would differ from a re-run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from reversi.endgame.solver import RootSolution, empties, solve_exact, solve_root
from reversi.errors import ReversiError
from reversi.game import rules
from reversi.game.bitboard import popcount
from reversi.game.state import State
from reversi.nn.features import legal_mask, masked_policy
from reversi.types import Action, pass_action

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence
    from pathlib import Path

    from reversi.search.evaluator import Evaluator

log = logging.getLogger(__name__)

__all__ = [
    "KEEPERS_PER_GAME",
    "MINING_LEVELS",
    "STAGES",
    "MiningError",
    "Puzzle",
    "Stage",
    "difficulty_of",
    "mine",
    "stage_for",
    "write_obf",
]


class MiningError(ReversiError):
    """Raised when mining cannot produce the curriculum it was asked for."""


# ---------------------------------------------------------------------------
# The curriculum
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Stage:
    """One rung of the progression: a band of empty-square counts, and a lesson.

    Called a *stage* rather than a level on purpose. The app already calls the
    six opponents Level 1 to Level 6, and a control panel that says "Level" about
    two unrelated things is a bug waiting to be filed as confusion. This
    repository has been bitten by exactly that before, when "Strong" was both a
    ladder rung and a thinking time.
    """

    number: int
    title: str
    teaches: str
    low: int
    high: int

    def contains(self, open_squares: int) -> bool:
        return self.low <= open_squares <= self.high


STAGES: tuple[Stage, ...] = (
    Stage(
        number=1,
        title="Count to the end",
        teaches="Few enough squares left to play the whole thing out in your head.",
        low=4,
        high=5,
    ),
    Stage(
        number=2,
        title="Who moves last",
        teaches=(
            "An odd number of empty squares in a region means you take the last one there, "
            "and a disc placed last can never be flipped back."
        ),
        low=6,
        high=7,
    ),
    Stage(
        number=3,
        title="Two regions at once",
        teaches="The same counting, split across separate holes in the board that interact.",
        low=8,
        high=9,
    ),
    Stage(
        number=4,
        title="Give discs away",
        teaches="Taking fewer discs now to keep hold of the moves that decide the ending.",
        low=10,
        high=11,
    ),
    Stage(
        number=5,
        title="Everything at once",
        teaches="Deep enough that no single rule finds the move for you.",
        low=12,
        high=13,
    ),
)

MINING_LEVELS: tuple[str, ...] = ("casual", "club", "strong")
"""Difficulty levels the mined games are played at.

Several rather than one, because a single opponent playing itself visits a narrow
slice of the game and the mistakes that make a position interesting are not in
it. Mixing strengths also produces the asymmetry a puzzle needs -- one side ahead
with a way to throw it away -- far more often than two equals do.
"""

KEEPERS_PER_GAME = 2
"""How many puzzles one game may contribute.

Positions from the same game share most of their board. Without a cap a handful
of games would supply the whole curriculum, and a player would meet the same
shape five times in a row while the file still looked varied from the outside.
"""


def stage_for(open_squares: int) -> Stage | None:
    """The stage a position belongs to, or ``None`` if it is outside every band."""
    for stage in STAGES:
        if stage.contains(open_squares):
            return stage
    return None


# ---------------------------------------------------------------------------
# One puzzle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Puzzle:
    """A winnable position, its exact answer table, and why it is worth asking."""

    stage: int
    state: State
    empties: int

    margins: dict[Action, int]
    """Exact final disc difference after each legal move, for the side to move."""
    best: int
    best_moves: tuple[Action, ...]
    winning_moves: tuple[Action, ...]
    principal_variation: tuple[Action, ...]

    edax_margins: dict[Action, int]
    """The same table under the convention that gives leftover empty squares to
    the winner. Not shipped to the app -- it is here so ``write_obf`` can emit a
    file Edax will check line by line, in Edax's own terms."""

    tempting_move: Action
    """The move the network's policy likes most."""
    tempting_margin: int
    trap_mass: float
    """How much of the network's probability sits on moves that lose."""
    greedy_falls: bool
    """True when *every* move that flips the most discs loses the game."""
    difficulty: int
    nodes: int

    @property
    def tempting_move_loses(self) -> bool:
        return self.tempting_margin <= 0

    def describe(self) -> str:
        trap = "network" if self.tempting_move_loses else ""
        if self.greedy_falls:
            trap = f"{trap}+greedy" if trap else "greedy"
        return (
            f"stage {self.stage}: {self.empties} empties, best {self.best:+d}, "
            f"{len(self.winning_moves)} of {len(self.margins)} moves win, "
            f"trap {self.trap_mass:.0%} ({trap}), difficulty {self.difficulty}"
        )


# ---------------------------------------------------------------------------
# Deriving difficulty
# ---------------------------------------------------------------------------

_NARROWNESS_WEIGHT = 0.45
_TRAP_WEIGHT = 0.35
_GREEDY_WEIGHT = 0.20


def difficulty_of(margins: dict[Action, int], policy: dict[Action, float], *, greedy: bool) -> int:
    """How hard this position is to get right, on a 0-100 scale.

    Derived from the solve and the network, never assigned, so that the order
    inside a stage is a consequence of the position rather than an opinion about
    it. Three ingredients, in descending order of how much they count:

    **How narrow the win is.** One winning move out of eight is a far harder ask
    than five out of eight, and it is exact rather than estimated. The dominant
    term, because it is the only one that measures the position itself.

    **How tempting the losing moves are.** The share of the network's probability
    resting on moves that throw the win away. A position where the natural move
    happens to be the right one is easy however few winning moves it has.

    **Whether grabbing discs loses.** A flat bonus, because it is a yes or no. It
    is weighted least despite teaching the most useful lesson: it says something
    about one particular beginner habit rather than about how hard the position
    is to read.

    An integer, so the shipped order cannot drift on a float comparison between
    two runs or two languages.
    """
    legal = len(margins)
    winning = sum(1 for margin in margins.values() if margin > 0)
    if legal < 2 or winning < 1:
        msg = (
            "difficulty is only defined for a winnable position with a choice; got "
            f"{legal} legal moves of which {winning} win"
        )
        raise MiningError(msg)

    narrowness = 1.0 - (winning - 1) / (legal - 1)
    trap_mass = sum(weight for move, weight in policy.items() if margins.get(move, 1) <= 0)
    score = (
        _NARROWNESS_WEIGHT * narrowness
        + _TRAP_WEIGHT * trap_mass
        + _GREEDY_WEIGHT * (1.0 if greedy else 0.0)
    )
    return round(100 * score)


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------


def _placements(state: State) -> list[Action]:
    """Legal moves that are actually moves, with a pass excluded."""
    skip = pass_action(state.size)
    return [action for action in rules.legal_actions(state) if action != skip]


def _max_flip_moves(state: State) -> list[Action]:
    """Every move that flips the most discs -- what a disc-grabber would consider.

    The whole set, not one of them. ``GreedyAgent`` breaks ties at random, so
    "grabbing discs loses here" is only a fact about the position when *no*
    tie-break saves it. Picking a single move would also make a committed
    artifact depend on a random draw, which it must not.
    """
    placements = _placements(state)
    if not placements:
        return []
    gains = {action: popcount(rules.flips(state, action)) for action in placements}
    most = max(gains.values())
    return [action for action in placements if gains[action] == most]


def _policy_over_legal(evaluator: Evaluator, state: State) -> dict[Action, float]:
    """The network's preference over the legal moves, as probabilities summing to 1.

    One forward pass and no search. What is wanted here is the move a player is
    drawn to before thinking, and that is the raw policy -- a search would find
    the right answer and stop being a model of the mistake.
    """
    logits, _ = evaluator.evaluate([state])
    probabilities = masked_policy(logits[0], legal_mask(state))
    return {action: float(probabilities[action]) for action in _placements(state)}


def _margin_after(state: State, action: Action, max_empties: int) -> int:
    """The exact result of one move, from the point of view of whoever played it."""
    child = rules.apply(state, action)
    return -solve_exact(child, max_empties=max_empties)


def _screen(
    state: State, evaluator: Evaluator, max_empties: int
) -> tuple[dict[Action, float], list[Action]] | None:
    """Decide whether a candidate is worth a full root solve.

    Returns the network's policy and the disc-grabbing moves when it is, so the
    caller does not pay for them twice, and ``None`` when it is not.

    The order of the questions is the whole point: the cheapest one that rejects
    the most candidates comes first. Most positions are simply not won by the
    side to move, and one search settles that.
    """
    if solve_exact(state, max_empties=max_empties) <= 0:
        return None

    policy = _policy_over_legal(evaluator, state)
    grabbing = _max_flip_moves(state)

    tempting = max(policy, key=lambda action: policy[action])
    if _margin_after(state, tempting, max_empties) <= 0:
        return policy, grabbing

    # The network sees through it. It is still a puzzle if disc-grabbing does
    # not -- but only when every tie-break among the grabbing moves loses too.
    if all(_margin_after(state, action, max_empties) <= 0 for action in grabbing):
        return policy, grabbing
    return None


def _build(
    stage: Stage,
    state: State,
    solution: RootSolution,
    edax: RootSolution,
    policy: dict[Action, float],
    grabbing: Sequence[Action],
) -> Puzzle:
    """Assemble a puzzle from the full solve.

    Every fact comes from ``solution``, including the ones the screen already
    looked at. The screen is a filter and nothing more; if its answers and the
    root solve's could ever disagree, the shipped file follows the one that
    examined every move.
    """
    tempting = max(policy, key=lambda action: policy[action])
    greedy_falls = all(solution.margins[action] <= 0 for action in grabbing)
    return Puzzle(
        stage=stage.number,
        state=state,
        empties=empties(state),
        margins=dict(solution.margins),
        best=solution.best,
        best_moves=solution.best_moves,
        winning_moves=solution.winning_moves,
        principal_variation=solution.principal_variation,
        edax_margins=dict(edax.margins),
        tempting_move=tempting,
        tempting_margin=solution.margins[tempting],
        trap_mass=sum(
            weight for move, weight in policy.items() if solution.margins.get(move, 1) <= 0
        ),
        greedy_falls=greedy_falls,
        difficulty=difficulty_of(solution.margins, policy, greedy=greedy_falls),
        nodes=solution.nodes,
    )


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------


def _play_one(evaluator: Evaluator, levels: tuple[str, str], seed: int, size: int) -> list[State]:
    """Play one game between two difficulty levels and return every position in it."""
    from reversi.difficulty.calibrate import DifficultyAgent
    from reversi.difficulty.levels import level_by_name

    rng = np.random.default_rng(seed)
    agents = tuple(DifficultyAgent(evaluator, level_by_name(name)) for name in levels)

    state = rules.initial_state(size)
    positions = [state]
    while not rules.is_terminal(state):
        state = rules.apply(state, agents[int(state.to_move)].select(state, rng))
        positions.append(state)
    return positions


def mine(
    evaluator: Evaluator,
    *,
    quota: int = 12,
    stages: Sequence[Stage] = STAGES,
    levels: Sequence[str] = MINING_LEVELS,
    board_size: int = 8,
    seed: int = 20260919,
    max_games: int = 600,
) -> list[Puzzle]:
    """Play games, keep the positions that are puzzles, and fill each stage.

    Per-stage quotas rather than a global top ``N``. A global ranking would
    quietly return a file of nothing but the hardest positions it found, which is
    the opposite of a progression -- and the hardest positions are also the
    deepest, so the file would be all stage 5 and cost the most to produce.

    Returns the puzzles already in presentation order: by stage, then by derived
    difficulty, easiest first. The app reads the file in order and never
    re-derives an ordering the generator was responsible for.
    """
    wanted = {stage.number: stage for stage in stages}
    kept: dict[int, list[Puzzle]] = {number: [] for number in wanted}
    seen: set[tuple[int, int, int]] = set()
    pairings = [(a, b) for a in levels for b in levels]
    games = 0
    screened_count = 0
    started = time.perf_counter()

    while games < max_games and any(len(kept[n]) < quota for n in wanted):
        pairing = pairings[games % len(pairings)]
        positions = _play_one(evaluator, pairing, seed + games, board_size)
        games += 1
        from_this_game = 0

        for state in positions:
            if from_this_game >= KEEPERS_PER_GAME:
                break
            open_squares = empties(state)
            stage = stage_for(open_squares)
            if stage is None or stage.number not in wanted or len(kept[stage.number]) >= quota:
                continue
            if len(_placements(state)) < 2:
                continue
            key = (state.black, state.white, int(state.to_move))
            if key in seen:
                continue
            seen.add(key)

            screened_count += 1
            screened = _screen(state, evaluator, open_squares)
            if screened is None:
                continue
            policy, grabbing = screened

            solution = solve_root(state, max_empties=open_squares)
            edax = solve_root(state, max_empties=open_squares, empties_to_winner=True)
            puzzle = _build(stage, state, solution, edax, policy, grabbing)
            kept[stage.number].append(puzzle)
            from_this_game += 1
            log.info("kept %s", puzzle.describe())

        # A line per game, whether or not it produced anything. Most games do
        # not, and a mining run that prints nothing for twenty minutes is
        # indistinguishable from one that has hung.
        filled = " ".join(f"s{number}:{len(kept[number])}/{quota}" for number in sorted(kept))
        log.info(
            "game %d: %s (%d positions screened, %.1f min elapsed)",
            games,
            filled,
            screened_count,
            (time.perf_counter() - started) / 60,
        )

    short = {number: len(found) for number, found in kept.items() if len(found) < quota}
    if short:
        detail = ", ".join(f"stage {number} has {count}" for number, count in sorted(short.items()))
        log.warning("quota of %d not reached after %d games: %s", quota, games, detail)

    ordered: list[Puzzle] = []
    for number in sorted(kept):
        # Difficulty first, then the empty count, then the board itself. The last
        # two are there so that two puzzles scoring the same land in the same
        # order on every run instead of following dictionary insertion.
        ordered.extend(
            sorted(
                kept[number],
                key=lambda p: (p.difficulty, p.empties, p.state.black, p.state.white),
            )
        )
    return ordered


# ---------------------------------------------------------------------------
# Evidence: the Edax problem file
# ---------------------------------------------------------------------------


def write_obf(destination: Path, puzzles: Sequence[Puzzle]) -> int:
    """Write the puzzles as an Edax problem file, and return the byte count.

    The ``.obf`` format carries a position and every move's exact score, which
    means ``wEdax -solve`` on this file does not merely re-solve the positions --
    it checks the answers and reports the ones it disagrees with. That turns the
    file into a standing cross-check against an outside engine rather than a
    dump.

    The scores written here are therefore in **Edax's** convention, where a game
    that ends with the board unfilled awards the leftover squares to the winner.
    Writing this project's numbers into a file with Edax's extension would
    produce a check that fails on correct answers.
    """
    from reversi.agents.edax import obf_string, square_name

    lines = []
    for puzzle in puzzles:
        scores = sorted(puzzle.edax_margins.items(), key=lambda item: (-item[1], item[0]))
        moves = "".join(f" {square_name(move)}:{margin:+d};" for move, margin in scores)
        lines.append(f"{obf_string(puzzle.state)};{moves}")

    text = "\n".join(lines) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Newlines held to LF rather than left to the platform. ``.gitattributes``
    # normalises the committed copy either way, but this file is also read
    # directly by Edax on the machine that wrote it, and a stray carriage return
    # at the end of a problem line is not something an outside parser has to
    # tolerate.
    destination.write_text(text, encoding="utf-8", newline="\n")
    return len(text.encode("utf-8"))
