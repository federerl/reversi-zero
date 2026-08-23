"""The gate: can a 4x4 agent learn Reversi from random weights? (S7, backlog T19)

This is the only test in the project that answers the question the whole project
is about. Everything else checks that a part behaves as specified; this one checks
that the parts together produce an agent that is *actually better* at the game
than it was when it started -- measured by playing, not by looking at a loss.

It runs the real `smoke4x4` profile end to end -- twelve generations of self-play
and training, about eight minutes on a laptop CPU -- and then plays 200
colour-balanced games against each opponent.

**If this fails, nothing downstream is trustworthy.** A rules bug, an inverted
value sign, a policy target that does not mean what the trainer thinks it means:
all of them let every other test pass while producing an agent that cannot beat a
coin flip. The plan says to stop all other work until this is green, and that is
the right instruction.

4x4 is a *fixture*, never a result. It exists because it fits in ten minutes on a
CPU, so the pipeline can be validated before an overnight GPU run is spent on it.
No number from this file belongs in the README.

**Two measured properties of the 4x4 board, worth knowing before reading the
thresholds.**

*Greedy is weaker than Random here.* On 8x8, Greedy beats Random about 63% of the
time. On 4x4 it is the other way round -- Greedy scores about 43%. The board is
small enough that grabbing discs early is close to self-defeating. So on this
board ">= 90% vs Random" is the demanding bar and ">= 65% vs Greedy" is nearly
free. Both are kept because both are in the specification, but only the first
means much here.

*White wins 4x4 with perfect play.* Proved exactly in
``tests/unit/test_solved_4x4.py`` -- the whole game is 3,306 positions. So a
trained agent scoring near 100% as white and lower as black is not lopsided, it
is correct: as black it is playing a theoretically lost game and can only win
when its opponent errs. An agent that scored *equally* with both colours on this
board would be the surprising one, and the colour test below is set accordingly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from reversi.agents import GreedyAgent, RandomAgent
from reversi.agents.az_agent import AZAgent
from reversi.arena import MatchResult, play_match
from reversi.config import Config, load_config
from reversi.obs.runmeta import RunPaths
from reversi.train.loop import run_training

pytestmark = [pytest.mark.slow, pytest.mark.timeout(2400)]

GAMES = 200
MINIMUM_VS_RANDOM = 0.90
MINIMUM_VS_GREEDY = 0.65
TIME_BUDGET_SECONDS = 900  # 10 min is the target; 15 allows for a loaded CI runner


@dataclass(slots=True)
class TrainedRun:
    config: Config
    paths: RunPaths
    seconds: float
    generations: int
    first_loss: float
    final_loss: float


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> TrainedRun:
    """Train once; every assertion below reads the same run."""
    config = load_config(Path("configs/smoke4x4.yaml"))
    paths = RunPaths(run_id="smoke4x4-gate", root=tmp_path_factory.mktemp("gate"))

    started = time.perf_counter()
    reports = run_training(config, paths)
    elapsed = time.perf_counter() - started

    assert reports, "the training loop produced no generations"
    return TrainedRun(
        config=config,
        paths=paths,
        seconds=elapsed,
        generations=len(reports),
        first_loss=reports[0].training["policy_loss"],
        final_loss=reports[-1].training["policy_loss"],
    )


def _match(trained: TrainedRun, opponent: RandomAgent | GreedyAgent) -> MatchResult:
    agent = AZAgent.from_checkpoint(
        trained.paths.checkpoints / "latest.pt",
        simulations=trained.config.mcts.n_simulations,
    )
    return play_match(
        agent,
        opponent,
        games=GAMES,
        board_size=trained.config.game.board_size,
        seed=20260823,
    )


# ===========================================================================
# The gate itself
# ===========================================================================


def test_the_agent_beats_a_random_opponent(trained: TrainedRun) -> None:
    """The demanding half of the gate on this board.

    Random is the floor. An agent that cannot beat it nine times in ten has not
    learned to play -- and since Random is also the anchor of the rating scale
    (0 Elo by definition), this is where every later number is measured from.
    """
    result = _match(trained, RandomAgent())
    assert result.score >= MINIMUM_VS_RANDOM, result.summary()


def test_the_agent_beats_a_greedy_opponent(trained: TrainedRun) -> None:
    """Greedy takes whichever move flips the most discs, which is the obvious
    wrong strategy: discs flip back, and grabbing them early costs mobility and
    corners. See the module docstring for why this bar is the easier one on 4x4.
    """
    result = _match(trained, GreedyAgent())
    assert result.score >= MINIMUM_VS_GREEDY, result.summary()


def test_the_training_fits_the_time_budget(trained: TrainedRun) -> None:
    """The budget is what makes this usable as a gate at all.

    A pipeline check nobody runs because it takes an hour is not a pipeline check.
    Ten minutes is the target; the assertion allows fifteen so a loaded CI runner
    does not produce a red build that says nothing about the code.
    """
    assert trained.seconds <= TIME_BUDGET_SECONDS, (
        f"training took {trained.seconds / 60:.1f} min for {trained.generations} generations"
    )


# ===========================================================================
# Diagnostics -- these say *why* if the gate fails
# ===========================================================================


def test_the_agent_wins_with_both_colours(trained: TrainedRun) -> None:
    """A large colour gap would mean an opening was learned rather than the game.

    Some gap is expected and correct: white wins 4x4 with perfect play, so black
    is defending a lost position. The floor is therefore set well below the
    headline bar -- the point is to catch a *collapse* with one colour, not to
    demand symmetry the game itself does not have.
    """
    result = _match(trained, RandomAgent())
    assert result.score_as_black >= 0.70, result.summary()
    assert result.score_as_white >= 0.70, result.summary()
    # And the gap should point the way the solved game says it must.
    assert result.score_as_white >= result.score_as_black - 0.05, result.summary()


def test_training_actually_reduced_the_loss(trained: TrainedRun) -> None:
    """A diagnostic, never evidence of strength.

    If the gate fails *and* this fails, the problem is in training. If the gate
    fails while this passes, the network is learning to predict a search that is
    itself broken -- look at the value sign first.
    """
    assert trained.final_loss < trained.first_loss


def test_the_trained_agent_beats_its_own_starting_point(trained: TrainedRun) -> None:
    """The cleanest statement of "it learned": same architecture, same search
    budget, same everything except twelve generations of self-play.

    An absolute threshold can be met by a lucky opponent match-up. This cannot.
    """
    from reversi.nn.evaluator import TorchEvaluator
    from reversi.nn.model import build
    from reversi.search.config import SearchConfig

    simulations = trained.config.mcts.n_simulations
    untrained = AZAgent(
        TorchEvaluator(build(trained.config.net, trained.config.game.board_size, seed=0)),
        SearchConfig(n_simulations=simulations),
        name="az-untrained",
    )
    trained_agent = AZAgent.from_checkpoint(
        trained.paths.checkpoints / "latest.pt", simulations=simulations
    )

    result = play_match(
        trained_agent,
        untrained,
        games=GAMES,
        board_size=trained.config.game.board_size,
        seed=7,
    )
    assert result.score > 0.60, result.summary()
