"""Check the Run 1 numbers quoted in docs/experiments.md against the report they cite.

The experiment log quotes a handful of results from Run 1's cross-generation
tournament: the score against Greedy at each generation, the score against
Minimax-d4, Minimax-d4's own score against Greedy, and a split of the Greedy
result by colour. All of them should be readable off ``docs/crossgen.json``,
which is the report that tournament wrote. This script reads them off and
prints each quoted number next to the number the report actually holds, so
any claim that the report does not support stands out.

Usage::

    python scripts/audit_run1_claims.py [docs/crossgen.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# What the log says, as (generation, claimed score vs Greedy, claimed score vs Minimax-d4).
QUOTED_BY_GENERATION = {
    "gen05": (0.90, 0.57),
    "gen20": (0.70, 0.87),
    "gen40": (0.70, 0.97),
    "gen60": (0.63, 1.00),
}
QUOTED_GREEDY_CI = (0.46, 0.78)
QUOTED_MINIMAX_D4_VS_GREEDY = 0.97
QUOTED_GEN60_VS_GREEDY_BY_COLOUR = {"as black": 0.90, "as white": 0.50}


def matchup(report: dict, a: str, b: str) -> tuple[dict, bool]:
    """The pairing between ``a`` and ``b``, and whether ``a`` was the first agent in it."""
    for row in report["matchups"]:
        if (row["a"], row["b"]) == (a, b):
            return row, True
        if (row["a"], row["b"]) == (b, a):
            return row, False
    msg = f"no pairing between {a} and {b} in the report"
    raise SystemExit(msg)


def score_for(row: dict, first: bool) -> tuple[float, float, float]:
    """Score for the agent asked about: overall, as black, as white.

    The report stores every number from its first agent's point of view. When the
    agent asked about was second, its score is one minus the first agent's, and
    its colours are the first agent's colours swapped.
    """
    if first:
        return row["score_a"], row["score_as_black"], row["score_as_white"]
    return 1 - row["score_a"], 1 - row["score_as_white"], 1 - row["score_as_black"]


def main(path: Path) -> int:
    report = json.loads(path.read_text())
    protocol = report["protocol"]
    sims = report["agent_specs"]["gen60"].get("simulations")
    print(f"report: {path}")
    print(
        f"protocol: {protocol['games_per_pair']} games per pairing, "
        f"{protocol['opening_plies']}-ply openings, seed {protocol['seed']}, "
        f"network entrants at {sims} simulations"
    )
    print()

    mismatches = 0

    def check(label: str, quoted: float, actual: float, tolerance: float = 0.01) -> None:
        nonlocal mismatches
        ok = abs(quoted - actual) <= tolerance
        mismatches += 0 if ok else 1
        flag = "ok " if ok else "MISMATCH"
        print(f"  {flag}  {label:<42} quoted {quoted:5.0%}   report {actual:5.1%}")

    print("score against Greedy and Minimax-d4, by generation:")
    for gen, (vs_greedy, vs_minimax) in QUOTED_BY_GENERATION.items():
        row, first = matchup(report, gen, "greedy")
        check(f"{gen} vs greedy", vs_greedy, score_for(row, first)[0])
        row, first = matchup(report, gen, "minimax-d4")
        check(f"{gen} vs minimax-d4", vs_minimax, score_for(row, first)[0])

    print("\nMinimax-d4 against Greedy:")
    row, first = matchup(report, "minimax-d4", "greedy")
    check("minimax-d4 vs greedy", QUOTED_MINIMAX_D4_VS_GREEDY, score_for(row, first)[0])

    print("\ngen60 against Greedy, the 95% interval (Wilson, from the report):")
    row, first = matchup(report, "gen60", "greedy")
    if first:
        low, high = row["ci_low"], row["ci_high"]
    else:
        low, high = 1 - row["ci_high"], 1 - row["ci_low"]
    check("gen60 vs greedy, interval low", QUOTED_GREEDY_CI[0], low)
    check("gen60 vs greedy, interval high", QUOTED_GREEDY_CI[1], high)

    print("\ngen60 against Greedy, split by colour:")
    _, as_black, as_white = score_for(row, first)
    check("gen60 vs greedy as black", QUOTED_GEN60_VS_GREEDY_BY_COLOUR["as black"], as_black)
    check("gen60 vs greedy as white", QUOTED_GEN60_VS_GREEDY_BY_COLOUR["as white"], as_white)

    print()
    if mismatches:
        print(f"{mismatches} quoted number(s) are not what the report holds.")
    else:
        print("every quoted number matches the report.")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/crossgen.json")
    raise SystemExit(main(target))
