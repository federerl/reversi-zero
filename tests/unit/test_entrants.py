"""Entrants are data, suites are line-ups, and the parallel round robin is the same tournament.

The arena's job is to rate agents fairly. These tests pin the plumbing around
that: that a competitor can be described in text, rebuilt in another process,
and written into a report in a way a reader can reproduce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reversi.arena.entrants import EntrantSpec, build_agent, describe_entrant, parse_entrant
from reversi.arena.suites import (
    baseline_entrants,
    checkpoint_label,
    list_generations,
    select_generations,
)
from reversi.arena.tournament import round_robin_parallel
from reversi.errors import ArenaError, ConfigError


@pytest.mark.parametrize(
    ("text", "kind", "name", "depth"),
    [
        ("random", "random", "random", None),
        ("greedy", "greedy", "greedy", None),
        ("minimax-d4", "minimax", "minimax-d4", 4),
        ("minimax4", "minimax", "minimax-d4", 4),
        ("minimax-d2", "minimax", "minimax-d2", 2),
        ("edax-l5", "edax", "edax-l5", 5),
        ("edax5", "edax", "edax-l5", 5),
    ],
)
def test_baselines_parse_from_their_names(
    text: str, kind: str, name: str, depth: int | None
) -> None:
    """The config's ``minimax4`` and the reports' ``minimax-d4`` are the same entrant."""
    spec = parse_entrant(text, default_simulations=50)
    assert spec.kind == kind
    assert spec.name == name
    assert spec.depth == depth
    assert not spec.needs_network


def test_a_network_entrant_is_named_and_given_a_budget(tmp_path: Path) -> None:
    weights = tmp_path / "gen_00060.pt"
    weights.write_bytes(b"not really weights")

    spec = parse_entrant(f"gen60={weights}@200", default_simulations=50)
    assert spec == EntrantSpec(name="gen60", kind="checkpoint", path=str(weights), simulations=200)

    unnamed = parse_entrant(str(weights), default_simulations=50)
    assert unnamed.name == "gen_00060"
    assert unnamed.simulations == 50


def test_search_settings_ride_along_after_a_semicolon(tmp_path: Path) -> None:
    """One network, several search constants: that is how c_puct gets tuned."""
    weights = tmp_path / "gen_00120.pt"
    weights.write_bytes(b"x")

    spec = parse_entrant(f"e1-c25={weights}@50;c_puct=2.5;fpu=0.1", default_simulations=1)
    assert spec.c_puct == 2.5
    assert spec.fpu_reduction == 0.1
    assert spec.simulations == 50
    assert describe_entrant(spec)["c_puct"] == 2.5

    with pytest.raises(ConfigError, match="unknown search option"):
        parse_entrant(f"x={weights};temperature=1", default_simulations=1)
    with pytest.raises(ConfigError, match="must be a number"):
        parse_entrant(f"x={weights};c_puct=lots", default_simulations=1)


def test_a_missing_network_file_is_refused_with_the_alternatives_named(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no network file"):
        parse_entrant(f"ghost={tmp_path / 'absent.pt'}", default_simulations=50)


def test_a_level_entrant_carries_its_rung_and_its_network(tmp_path: Path) -> None:
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"x")
    spec = parse_entrant(f"level:casual={weights}", default_simulations=50)
    assert spec.kind == "level"
    assert spec.level == "casual"
    assert spec.path == str(weights)
    assert spec.needs_network


def test_baselines_build_without_torch_and_keep_their_names() -> None:
    agents = [build_agent(spec) for spec in baseline_entrants(("random", "greedy", "minimax4"))]
    assert [a.name for a in agents] == ["random", "greedy", "minimax-d4"]


def test_the_report_description_lets_a_reader_reproduce_the_entrant(tmp_path: Path) -> None:
    """A minimax entry records the weights fingerprint, a checkpoint its file and
    generation, because a rating without those is a number nobody can check."""
    minimax = describe_entrant(parse_entrant("minimax-d4", default_simulations=1))
    assert minimax["kind"] == "minimax"
    assert minimax["depth"] == 4
    assert len(minimax["weights"]) == 16

    weights = tmp_path / "gen_00060.pt"
    weights.write_bytes(b"x")
    weights.with_suffix(".json").write_text(
        json.dumps({"generation": 60, "run_id": "some-run"}), encoding="utf-8"
    )
    net = describe_entrant(parse_entrant(f"gen60={weights}@50", default_simulations=1))
    assert net == {
        "kind": "alphazero",
        "checkpoint": "gen_00060.pt",
        "generation": 60,
        "run_id": "some-run",
        "simulations": 50,
    }


def test_checkpoint_labels_follow_the_reports_convention(tmp_path: Path) -> None:
    assert checkpoint_label(Path("gen_00005.pt")) == "gen05"
    assert checkpoint_label(Path("gen_00120.pt")) == "gen120"

    latest = tmp_path / "latest.pt"
    latest.write_bytes(b"x")
    latest.with_suffix(".json").write_text(json.dumps({"generation": 37}), encoding="utf-8")
    assert checkpoint_label(latest) == "gen37"

    assert checkpoint_label(tmp_path / "reversi-8x8-gen60.pt") == "reversi-8x8-gen60"


def test_generations_are_listed_from_disk_and_spread_evenly(tmp_path: Path) -> None:
    for g in (5, 10, 15, 20, 25, 30, 35, 40, 41, 42):
        (tmp_path / f"gen_{g:05d}.pt").write_bytes(b"x")
    (tmp_path / "latest.pt").write_bytes(b"x")
    (tmp_path / "current.pt").write_bytes(b"x")

    available = list_generations(tmp_path)
    assert available == [5, 10, 15, 20, 25, 30, 35, 40, 41, 42]

    chosen = select_generations(available, max_checkpoints=4)
    assert chosen[0] == 5 and chosen[-1] == 42
    assert len(chosen) == 4
    assert select_generations([5, 10], max_checkpoints=6) == [5, 10]
    with pytest.raises(ArenaError):
        select_generations(available, max_checkpoints=1)


def test_the_parallel_round_robin_refuses_a_field_without_its_anchor() -> None:
    field = baseline_entrants(("greedy", "minimax-d1"))
    with pytest.raises(ArenaError, match="anchor"):
        round_robin_parallel(field, games_per_pair=2, board_size=4, seed=1, opening_plies=0)


def test_the_parallel_round_robin_gives_the_same_result_however_it_is_split() -> None:
    """Seeds are derived per pairing, so one process and two must agree game for game."""
    field = baseline_entrants(("random", "greedy", "minimax-d1"))

    def play(workers: int) -> list[tuple[str, str, int, int, int]]:
        result = round_robin_parallel(
            field,
            games_per_pair=4,
            board_size=4,
            seed=20260904,
            workers=workers,
            opening_plies=0,
            bootstrap=0,
        )
        assert {r.name for r in result.ratings.sorted()} == {"random", "greedy", "minimax-d1"}
        assert result.ratings.by_name()["random"].elo == 0.0
        return sorted((m.agent_a, m.agent_b, m.wins, m.losses, m.draws) for m in result.matches)

    in_process = play(1)
    across_processes = play(2)
    assert len(in_process) == 3
    assert in_process == across_processes
