"""How long an exact solve actually takes, per empty square.

The solver has no depth limit, so "can it solve this?" is entirely a question of
how many squares are still empty -- and the cost roughly triples with each one.
That curve decides two things the rest of the feature is built on: how deep the
puzzle curriculum can go, and whether mining thousands of candidates is an
afternoon or a week.

Run it rather than guess at it:

    uv run python bench/endgame_bench.py

Positions come from real play, not from random boards. A random position with
twelve empty squares tends to have a strange, sparse shape that searches far
faster than anything a game actually reaches, which would make the numbers
flattering and useless.
"""

from __future__ import annotations

import random
import time

from reversi.endgame.solver import empties, solve_root
from reversi.game import rules
from reversi.game.state import State, initial_state
from reversi.types import Action


def positions_with(open_squares: int, *, count: int, seed: int, board_size: int = 8) -> list[State]:
    """Play random games and keep positions with exactly this many empties.

    Only positions where the mover has a real choice are kept: a forced move has
    nothing to solve, and including them would quietly lower the average.
    """
    rng = random.Random(seed)
    found: list[State] = []
    seen: set[tuple[int, int, int]] = set()

    while len(found) < count:
        state = initial_state(board_size)
        while not rules.is_terminal(state):
            if empties(state) == open_squares:
                placements = rules.legal_placements(state)
                key = (state.black, state.white, int(state.to_move))
                if placements and key not in seen and bin(placements).count("1") > 1:
                    seen.add(key)
                    found.append(state)
                    break
            actions: list[Action] = rules.legal_actions(state)
            state = rules.apply(state, rng.choice(actions))

    return found


def main() -> None:
    print(f"{'empties':>8}{'positions':>11}{'median s':>11}{'worst s':>10}{'knodes':>10}")
    for open_squares in (8, 10, 11, 12, 13, 14):
        # Fewer samples as it gets expensive: the point is the shape of the
        # curve, and a slow row does not need thirty readings to be believed.
        count = 20 if open_squares <= 11 else 8
        states = positions_with(open_squares, count=count, seed=20260914 + open_squares)

        times: list[float] = []
        nodes = 0
        for state in states:
            started = time.perf_counter()
            solution = solve_root(state, max_empties=open_squares)
            times.append(time.perf_counter() - started)
            nodes += solution.nodes

        times.sort()
        median = times[len(times) // 2]
        print(
            f"{open_squares:>8}{len(states):>11}{median:>11.3f}"
            f"{times[-1]:>10.3f}{nodes / len(states) / 1000:>10.1f}"
        )


if __name__ == "__main__":
    main()
