"""Rating a run from inside the run: a quick match every few generations.

Training loss says nothing about strength, and until now nothing in the loop
measured strength at all. A run produced a hundred checkpoints and the question
"which one is best" was answered afterwards, by hand, or not at all.

This plays the newest checkpoint against a few fixed opponents every
``arena.every_n_generations`` generations and fits one rating to the result. The
number is written into the checkpoint's sidecar as ``elo_estimate``, logged to the
``arena`` metrics stream, and used to keep ``best.pt`` pointing at the strongest
checkpoint so far.

**What the number is, and is not.** It is a within-run, low-precision estimate:
a few dozen games at a modest search budget against Random, Greedy and a shallow
Minimax. It is enough to draw a curve and to pick ``best``, and it is anchored at
Random = 0 like every other rating in this project. It is **not** the
cross-generation table that ``reversi arena --suite crossgen`` produces, and the
two scales agree only on the anchor. Once the agent beats all three opponents
nearly every game the estimate saturates; that is the signal to add a stronger
opponent to ``arena.quick_opponents``, or to accept that the quick match has told
you what it can.

**Why it reuses the arena.** The match protocol, the seeding, the fairness
checks and the rating fit are the arena's, called through ``round_robin_parallel``
on the checkpoint file that was just written. The pairings run in CPU processes:
a tournament asks the network about one position at a time, and a GPU is barely
faster than a CPU core at that. Same protocol as every published number here, so
the quick estimate and the real table can be compared in shape even though they
cannot be compared in scale.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reversi.arena.entrants import EntrantSpec, parse_entrant
from reversi.arena.tournament import round_robin_parallel
from reversi.config import Config
from reversi.seeding import derive_seed

__all__ = ["QuickEval", "quick_evaluate"]

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QuickEval:
    """One quick rating of one checkpoint."""

    generation: int
    elo_estimate: float
    """Bradley-Terry rating against the quick opponents, anchored at Random = 0."""
    scores: dict[str, float] = field(default_factory=dict)
    """The agent's score against each opponent: wins plus half the draws, per game."""
    games: int = 0
    simulations: int = 0
    seconds: float = 0.0

    def as_metrics(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "elo_estimate": self.elo_estimate,
            "games_per_opponent": self.games,
            "simulations": self.simulations,
            "seconds": self.seconds,
        }
        for name, score in self.scores.items():
            out[f"score_vs_{name.replace('-', '_')}"] = score
        return out


def quick_evaluate(
    checkpoint: Path,
    *,
    config: Config,
    generation: int,
) -> QuickEval:
    """Rate the checkpoint at ``checkpoint`` against ``config.arena.quick_opponents``.

    Every pairing among the agent and the opponents is played, not just the
    agent's, so the rating fit has a connected graph to work with. The opponents'
    games against each other are cheap: none of them uses a network.
    """
    arena = config.arena
    agent = EntrantSpec(
        name=f"gen{generation:02d}",
        kind="checkpoint",
        path=str(checkpoint),
        simulations=arena.quick_simulations,
    )
    opponents = [parse_entrant(name, default_simulations=1) for name in arena.quick_opponents]
    names = [o.name for o in opponents]
    anchor = "random" if "random" in names else names[0]

    started = time.perf_counter()
    result = round_robin_parallel(
        [agent, *opponents],
        games_per_pair=arena.quick_games,
        board_size=config.game.board_size,
        # One seed per generation: the same checkpoint rated twice gets the same
        # games, and consecutive generations get different openings.
        seed=derive_seed(config.seed, "quick", generation),
        workers=arena.quick_workers,
        opening_plies=arena.opening_plies,
        anchor=anchor,
        bootstrap=0,
        device="cpu",
        scope="quick",
    )

    scores: dict[str, float] = {}
    for match in result.matches:
        if match.agent_a == agent.name:
            scores[match.agent_b] = match.score
        elif match.agent_b == agent.name:
            scores[match.agent_a] = 1.0 - match.score

    rating = result.ratings.by_name()[agent.name].elo
    log.info(
        "quick rating at generation %d: %+.0f Elo vs %s (%d games each, %d simulations, %.0fs)",
        generation,
        rating,
        ", ".join(f"{name} {score:.0%}" for name, score in scores.items()),
        arena.quick_games,
        arena.quick_simulations,
        time.perf_counter() - started,
    )
    return QuickEval(
        generation=generation,
        elo_estimate=float(rating),
        scores=scores,
        games=arena.quick_games,
        simulations=arena.quick_simulations,
        seconds=time.perf_counter() - started,
    )
