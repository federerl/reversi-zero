"""Configuration validation (backlog T03, criterion T27).

The point of these tests is that a misconfiguration fails *at startup with the
offending field named*, rather than silently wasting a night of GPU time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reversi.config import Config, load_config, parse_override
from reversi.errors import ConfigError

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
SHIPPED_PROFILES = ["smoke4x4.yaml", "dev8x8.yaml", "full8x8.yaml"]


# ---------------------------------------------------------------------------
# Shipped profiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", SHIPPED_PROFILES)
def test_shipped_profiles_load(profile: str) -> None:
    config = load_config(CONFIG_DIR / profile)
    assert config.game.board_size in (4, 6, 8)
    assert config.sha256


def test_base_alone_loads() -> None:
    assert load_config().name == "dev8x8"


def test_profile_overrides_base() -> None:
    smoke = load_config(CONFIG_DIR / "smoke4x4.yaml")
    assert smoke.game.board_size == 4
    assert smoke.net.channels == 16
    # Inherited from base rather than restated in the profile.
    assert smoke.mcts.c_puct == 1.5


def test_full_profile_uses_sgd_and_larger_net() -> None:
    full = load_config(CONFIG_DIR / "full8x8.yaml")
    assert full.train.optimizer == "sgd"
    assert (full.net.n_blocks, full.net.channels) == (6, 64)
    assert full.selfplay.n_workers == 6, "sized for the >=8 core cluster allocation"


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("size", "policy", "max_plies"), [(4, 17, 12), (6, 37, 32), (8, 65, 60)])
def test_derived_sizes(size: int, policy: int, max_plies: int) -> None:
    config = Config.model_validate({"game": {"board_size": size}, "mcts": {"temp_moves": 2}})
    assert config.game.policy_size == policy
    assert config.game.max_plies == max_plies
    assert config.game.n_squares == size * size


# ---------------------------------------------------------------------------
# Rejection of bad values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected_fragment"),
    [
        ({"game": {"board_size": 5}}, "board_size"),
        ({"mcts": {"n_simulations": 0}}, "n_simulations"),
        ({"mcts": {"n_simulations": -1}}, "n_simulations"),
        ({"mcts": {"dirichlet_eps": 1.5}}, "dirichlet_eps"),
        ({"mcts": {"dirichlet_eps": -0.1}}, "dirichlet_eps"),
        ({"mcts": {"c_puct": 0.0}}, "c_puct"),
        ({"train": {"optimizer": "rmsprop"}}, "optimizer"),
        ({"train": {"batch_size": 0}}, "batch_size"),
        ({"arena": {"games": 201}}, "games"),
        ({"selfplay": {"n_workers": 0}}, "n_workers"),
    ],
)
def test_invalid_values_are_rejected(payload: dict, expected_fragment: str) -> None:
    with pytest.raises(ValueError, match=expected_fragment):
        Config.model_validate(payload)


def test_unknown_key_is_an_error() -> None:
    """A typo'd hyperparameter must not be silently ignored."""
    with pytest.raises(ValueError, match="n_simulation"):
        Config.model_validate({"mcts": {"n_simulation": 100}})


def test_unknown_top_level_key_is_an_error() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        Config.model_validate({"learning_rate": 0.1})


# ---------------------------------------------------------------------------
# Cross-field invariants
# ---------------------------------------------------------------------------


def test_temp_moves_must_be_less_than_max_plies() -> None:
    with pytest.raises(ValueError, match="temp_moves"):
        Config.model_validate({"game": {"board_size": 4}, "mcts": {"temp_moves": 12}})


def test_replay_window_must_hold_a_batch() -> None:
    with pytest.raises(ValueError, match="replay.window"):
        Config.model_validate({"replay": {"window": 64}, "train": {"batch_size": 512}})


def test_workers_must_not_outnumber_games() -> None:
    with pytest.raises(ValueError, match="games_per_generation"):
        Config.model_validate({"selfplay": {"games_per_generation": 2, "n_workers": 8}})


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


def test_parse_override_nests_and_types() -> None:
    assert parse_override("mcts.n_simulations=200") == {"mcts": {"n_simulations": 200}}
    assert parse_override("train.lr=1e-3") == {"train": {"lr": 0.001}}
    assert parse_override("train.symmetry_aug=false") == {"train": {"symmetry_aug": False}}
    assert parse_override("name=probe") == {"name": "probe"}


@pytest.mark.parametrize("bad", ["nonsense", "=5", ""])
def test_parse_override_rejects_malformed(bad: str) -> None:
    with pytest.raises(ConfigError):
        parse_override(bad)


def test_overrides_apply_last() -> None:
    config = load_config(
        CONFIG_DIR / "smoke4x4.yaml",
        overrides=["mcts.n_simulations=99", "train.lr=0.05"],
    )
    assert config.mcts.n_simulations == 99
    assert config.train.lr == 0.05
    assert config.game.board_size == 4


def test_override_that_breaks_an_invariant_is_rejected() -> None:
    with pytest.raises(ConfigError, match="temp_moves"):
        load_config(CONFIG_DIR / "smoke4x4.yaml", overrides=["mcts.temp_moves=50"])


def test_missing_profile_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(CONFIG_DIR / "does-not-exist.yaml")


# ---------------------------------------------------------------------------
# Identity and immutability
# ---------------------------------------------------------------------------


def test_sha256_is_stable_and_sensitive() -> None:
    a = load_config(CONFIG_DIR / "dev8x8.yaml")
    b = load_config(CONFIG_DIR / "dev8x8.yaml")
    c = load_config(CONFIG_DIR / "dev8x8.yaml", overrides=["seed=2"])
    assert a.sha256 == b.sha256
    assert a.sha256 != c.sha256


def test_config_is_frozen() -> None:
    config = load_config()
    with pytest.raises(ValueError, match="frozen|immutable"):
        config.seed = 5  # type: ignore[misc]


def test_yaml_round_trip() -> None:
    original = load_config(CONFIG_DIR / "full8x8.yaml")
    import yaml

    restored = Config.model_validate(yaml.safe_load(original.to_yaml()))
    assert restored.sha256 == original.sha256
