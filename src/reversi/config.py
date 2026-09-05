"""Typed, validated configuration.

Design notes
------------
* Every model sets ``extra="forbid"``. A typo'd hyperparameter is an error at
  startup with the offending field path, not a silently ignored key that wastes
  a night of GPU time.
* Every model is frozen. Config is read-only once loaded; anything that wants a
  variation makes a copy via ``model_copy(update=...)``.
* Defaults in code are the 8x8 *development* values, so ``Config()`` works in
  tests without touching the filesystem. The shipped YAML profiles state every
  value explicitly anyway -- an experiment record that depends on code defaults
  is not reproducible.

Loading order: ``configs/base.yaml`` -> profile YAML -> ``--set`` CLI overrides.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from reversi.errors import ConfigError

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, validate_default=True)

VALID_BOARD_SIZES = (4, 6, 8)


class _Base(BaseModel):
    model_config = _MODEL_CONFIG


class GameConfig(_Base):
    """Board geometry. 4x4 exists to validate the pipeline; 8x8 is the product."""

    board_size: int = Field(default=8, description="4, 6, or 8. 8 is the headline result.")

    @property
    def n_squares(self) -> int:
        return self.board_size * self.board_size

    @property
    def policy_size(self) -> int:
        """Squares plus one PASS entry."""
        return self.n_squares + 1

    @property
    def max_plies(self) -> int:
        """Upper bound on placements in a game: every initially-empty square."""
        return self.n_squares - 4

    @model_validator(mode="after")
    def _check_board_size(self) -> GameConfig:
        if self.board_size not in VALID_BOARD_SIZES:
            msg = f"game.board_size must be one of {VALID_BOARD_SIZES}, got {self.board_size}"
            raise ValueError(msg)
        return self


class NetConfig(_Base):
    """Policy-value ResNet shape.

    Capacity is deliberately small: at this scale inference *latency*, not FLOPs,
    bounds self-play throughput, and games/hour is the scarce resource.
    """

    n_blocks: int = Field(default=4, ge=1, le=20)
    channels: int = Field(default=48, ge=8, le=256)
    value_hidden: int = Field(default=64, ge=8, le=512)
    ownership: bool = Field(
        default=False,
        description="Add a head that predicts who ends up owning each square. "
        "Trained only if train.ownership_loss_weight > 0; never used at play time.",
    )


class MCTSConfig(_Base):
    """PUCT search parameters.

    ``dirichlet_eps`` MUST be 0.0 outside self-play. The arena runner and the API
    engine service each assert this at construction (contract C7).
    """

    n_simulations: int = Field(default=100, ge=1)
    c_puct: float = Field(default=1.5, gt=0.0)
    fpu_reduction: float = Field(
        default=0.25,
        ge=0.0,
        description="First-play-urgency reduction for unvisited children. "
        "Using 0 here (i.e. Q=0 for unvisited) makes early search behave "
        "breadth-first -- a classic silent regression.",
    )
    dirichlet_alpha: float = Field(default=1.0, gt=0.0, description="~10 / mean branching factor")
    dirichlet_eps: float = Field(default=0.25, ge=0.0, le=1.0)
    temp_moves: int = Field(
        default=12,
        ge=0,
        description="Plies played at temp_init before switching to argmax. "
        "This governs the move PLAYED only; the stored policy target is always "
        "the raw visit distribution at tau=1 (contract C4).",
    )
    temp_init: float = Field(default=1.0, ge=0.0)


class SelfPlayConfig(_Base):
    """Self-play generation sizing.

    ``games_per_generation`` is tuned to wall-clock, not to a round number: with
    an 8 hour job ceiling and requeue-on-preemption, a generation of <=20 minutes
    bounds the worst-case loss from a kill to <=20 minutes.
    """

    games_per_generation: int = Field(default=800, ge=1)
    n_workers: int = Field(
        default=4, ge=1, description="Independent processes; no IPC between them"
    )
    games_in_flight: int = Field(
        default=32,
        ge=1,
        description="Games advanced in lockstep inside ONE worker, so that all "
        "their MCTS leaves batch into a single forward pass.",
    )
    max_generations: int = Field(default=15, ge=1)


class ReplayConfig(_Base):
    """Sliding-window replay buffer over per-generation .npz shards."""

    window: int = Field(default=100_000, ge=1, description="Positions retained for sampling")
    per_gen_cap_factor: float = Field(
        default=2.0,
        gt=0.0,
        description="Max share of a sampled batch one generation may contribute, "
        "as a multiple of its fair share. Stops one fast generation dominating.",
    )
    retain_shards: int = Field(default=30, ge=1, description="Shard files kept on disk")


class TrainConfig(_Base):
    """Optimisation.

    SGD for the full 8x8 run (better generalisation, AlphaZero-lineage default);
    AdamW for smoke/dev where iteration speed beats final quality.
    """

    steps_per_generation: int = Field(default=400, ge=1)
    batch_size: int = Field(default=512, ge=1)
    optimizer: Literal["sgd", "adamw"] = "adamw"
    lr: float = Field(default=1e-3, gt=0.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0, description="SGD only")
    weight_decay: float = Field(default=1e-4, ge=0.0)
    warmup_steps: int = Field(default=200, ge=0)
    lr_floor_divisor: float = Field(default=20.0, ge=1.0, description="Cosine decays to lr/this")
    grad_clip: float = Field(default=5.0, gt=0.0, description="Global-norm clip")
    value_loss_weight: float = Field(default=1.0, ge=0.0)
    ownership_loss_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight on the ownership head's loss. 0 disables the term. "
        "Needs net.ownership.",
    )
    symmetry_aug: bool = Field(
        default=True,
        description="Apply a random one of the 8 dihedral symmetries per sample "
        "at sampling time (not at storage time): 8x less disk, and every epoch "
        "sees a different orientation.",
    )
    checkpoint_every_steps: int = Field(
        default=200, ge=1, description="Mid-generation checkpoint cadence, for cheap resume"
    )


class ArenaConfig(_Base):
    """Evaluation protocol.

    Fairness is structural, not incidental: every matchup uses a seeded random
    opening book, each opening played twice with colours swapped, and exploration
    noise disabled.
    """

    every_n_generations: int = Field(default=5, ge=1)
    games: int = Field(
        default=200, ge=2, description="Per matchup; must be even for colour balance"
    )
    opening_plies: int = Field(
        default=4,
        ge=0,
        description="Random plies used to diversify openings. Without this, "
        "deterministic agents would replay one identical game every time.",
    )
    min_legal_after_opening: int = Field(default=2, ge=1)
    baselines: tuple[str, ...] = Field(default=("random", "greedy", "minimax4"))
    bootstrap_resamples: int = Field(default=1000, ge=100)

    @model_validator(mode="after")
    def _check_even(self) -> ArenaConfig:
        if self.games % 2 != 0:
            msg = f"arena.games must be even so colours split 50/50, got {self.games}"
            raise ValueError(msg)
        return self


class WandbConfig(_Base):
    """Weights & Biases is strictly optional; no core workflow may require it."""

    enabled: bool = False
    project: str = "reversi-zero"
    entity: str | None = None


class ObsConfig(_Base):
    """Observability. JSONL is the source of truth; TensorBoard mirrors scalars."""

    tensorboard: bool = True
    resource_sample_seconds: float = Field(default=1.0, gt=0.0)
    diagnostic_positions: int = Field(
        default=512,
        ge=0,
        description="Size of the fixed held-out set used for policy entropy and "
        "value-calibration metrics.",
    )
    wandb: WandbConfig = WandbConfig()


class Config(_Base):
    """Fully resolved run configuration."""

    name: str = Field(default="dev8x8", description="Profile name; appears in run_id")
    seed: int = Field(default=1337, ge=0)
    run_root: Path = Field(
        default=Path("runs"),
        description="Overridden on the cluster via the RZ_RUN_ROOT environment variable",
    )

    game: GameConfig = GameConfig()
    net: NetConfig = NetConfig()
    mcts: MCTSConfig = MCTSConfig()
    selfplay: SelfPlayConfig = SelfPlayConfig()
    replay: ReplayConfig = ReplayConfig()
    train: TrainConfig = TrainConfig()
    arena: ArenaConfig = ArenaConfig()
    obs: ObsConfig = ObsConfig()

    # ---- cross-field invariants -------------------------------------------

    @model_validator(mode="after")
    def _check_cross_field(self) -> Config:
        if self.mcts.temp_moves >= self.game.max_plies:
            msg = (
                f"mcts.temp_moves ({self.mcts.temp_moves}) must be less than the "
                f"maximum number of placements for a {self.game.board_size}x"
                f"{self.game.board_size} board ({self.game.max_plies}); otherwise "
                "every move is sampled and no game ever plays to strength."
            )
            raise ValueError(msg)

        if self.replay.window < self.train.batch_size:
            msg = (
                f"replay.window ({self.replay.window}) must be >= train.batch_size "
                f"({self.train.batch_size}); a batch cannot be drawn from a smaller window."
            )
            raise ValueError(msg)

        if self.selfplay.games_per_generation < self.selfplay.n_workers:
            msg = (
                f"selfplay.games_per_generation ({self.selfplay.games_per_generation}) "
                f"must be >= selfplay.n_workers ({self.selfplay.n_workers}); "
                "otherwise some workers would have no games to play."
            )
            raise ValueError(msg)

        if self.train.ownership_loss_weight > 0.0 and not self.net.ownership:
            msg = (
                f"train.ownership_loss_weight is {self.train.ownership_loss_weight} but "
                "net.ownership is false: there is no head to train. Set net.ownership: "
                "true, or the weight to 0."
            )
            raise ValueError(msg)

        return self

    # ---- derived / serialisation ------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=True, default_flow_style=False)

    @property
    def sha256(self) -> str:
        """Stable hash of the resolved config, recorded in every checkpoint."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ===========================================================================
# Loading
# ===========================================================================


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    out = dict(base)
    for key, value in override.items():
        existing = out.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            out[key] = _deep_merge(existing, value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        msg = f"config file not found: {path}"
        raise ConfigError(msg)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - exercised via test fixture
        msg = f"{path} is not valid YAML: {exc}"
        raise ConfigError(msg) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        msg = f"{path} must contain a YAML mapping at the top level, got {type(loaded).__name__}"
        raise ConfigError(msg)
    return loaded


def parse_override(item: str) -> dict[str, Any]:
    """Turn ``"mcts.n_simulations=200"`` into ``{"mcts": {"n_simulations": 200}}``.

    Values are parsed as YAML scalars, so ``true``, ``3``, ``1e-3``, and
    ``[a, b]`` all behave as expected.
    """
    if "=" not in item:
        msg = f"override {item!r} must be of the form path.to.field=value"
        raise ConfigError(msg)
    path, _, raw = item.partition("=")
    path = path.strip()
    if not path:
        msg = f"override {item!r} has an empty field path"
        raise ConfigError(msg)
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        msg = f"could not parse value in override {item!r}: {exc}"
        raise ConfigError(msg) from exc

    # PyYAML implements YAML 1.1, whose float resolver requires a decimal point
    # in exponent notation -- so `--set train.lr=1e-3` would otherwise arrive as
    # the string "1e-3". Recover the obvious numeric intent.
    if isinstance(value, str):
        value = _coerce_numeric(value)

    nested: dict[str, Any] = {}
    cursor = nested
    parts = path.split(".")
    for part in parts[:-1]:
        child: dict[str, Any] = {}
        cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value
    return nested


def _coerce_numeric(text: str) -> str | int | float:
    """Best-effort int/float recovery for values YAML 1.1 leaves as strings."""
    stripped = text.strip()
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return text


def load_config(
    profile: Path | str | None = None,
    *,
    base: Path | str | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load ``base.yaml``, layer a profile on top, then apply ``--set`` overrides.

    Raises ``ConfigError`` with the offending field path for any invalid value.
    """
    merged: dict[str, Any] = {}

    base_path = Path(base) if base is not None else _default_base_path()
    if base_path.is_file():
        merged = _deep_merge(merged, _read_yaml(base_path))

    if profile is not None:
        merged = _deep_merge(merged, _read_yaml(Path(profile)))

    for item in overrides or []:
        merged = _deep_merge(merged, parse_override(item))

    try:
        return Config.model_validate(merged)
    except ValueError as exc:
        msg = f"invalid configuration: {exc}"
        raise ConfigError(msg) from exc


def _default_base_path() -> Path:
    """Locate ``configs/base.yaml`` relative to the repository root."""
    return Path(__file__).resolve().parents[2] / "configs" / "base.yaml"
