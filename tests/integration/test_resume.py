"""Killing a training run and picking it back up (test matrix T24, S8).

The claim being tested is the one that decides whether an overnight run is worth
starting at all: **a run stopped at an arbitrary moment loses at most one
generation, and never resumes from something corrupt.**

Two ways a run stops, and both are covered here:

* **Politely** -- a signal arrives, the loop finishes the generation it is in and
  exits with everything on disk complete. This is what a scheduler warning or a
  first Ctrl-C should produce.
* **Violently** -- the process is killed outright, mid-write, with no chance to
  clean up. This is what a reboot, an OOM kill, or a second Ctrl-C produces, and
  it is the case that actually matters, because it is the one that leaves torn
  files behind.

The violent case is tested by really killing a real subprocess, not by simulating
it. A mock cannot leave a half-written file, and half-written files are the whole
problem.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from reversi.ckpt import CheckpointManager
from reversi.ckpt.meta import checkpoint_name
from reversi.config import Config
from reversi.data.shards import Manifest
from reversi.obs.runmeta import RunPaths
from reversi.obs.signals import StopFlag, cooperative_stop
from reversi.train.loop import run_training

pytestmark = pytest.mark.timeout(600)


@pytest.fixture
def paths(tmp_path: Path) -> RunPaths:
    return RunPaths(run_id="resume-test", root=tmp_path / "run")


def manager_for(paths: RunPaths, config: Config) -> CheckpointManager:
    return CheckpointManager(paths.checkpoints, run_id=paths.run_id, config_sha256=config.sha256)


# ===========================================================================
# The stop flag itself
# ===========================================================================


def test_a_stop_flag_starts_clear_and_latches() -> None:
    flag = StopFlag()
    assert not flag
    assert not flag()

    flag.request("SIGTERM")

    assert flag
    assert flag()
    assert flag.reason == "SIGTERM"


def test_the_first_reason_is_the_one_remembered() -> None:
    flag = StopFlag()
    flag.request("SIGTERM")
    flag.request("SIGINT")
    assert flag.reason == "SIGTERM"


def test_installing_handlers_restores_them_afterwards() -> None:
    """A caller's own Ctrl-C behaviour must not be left broken."""
    import signal

    before = signal.getsignal(signal.SIGINT)
    with cooperative_stop() as flag:
        assert not flag
    assert signal.getsignal(signal.SIGINT) is before


def test_handlers_are_restored_even_when_the_body_raises() -> None:
    import signal

    before = signal.getsignal(signal.SIGINT)
    with pytest.raises(RuntimeError), cooperative_stop():
        msg = "boom"
        raise RuntimeError(msg)
    assert signal.getsignal(signal.SIGINT) is before


def test_a_signal_stops_the_loop_between_generations(smoke_config: Config, paths: RunPaths) -> None:
    """The polite path, end to end: the flag is what the loop actually checks."""
    flag = StopFlag()
    seen = {"generations": 0}

    def should_stop() -> bool:
        seen["generations"] += 1
        if seen["generations"] == 3:
            flag.request("SIGTERM")
        return flag.requested

    reports = run_training(smoke_config, paths, generations=5, should_stop=should_stop)

    assert [r.generation for r in reports] == [1, 2]
    # Everything it did produce is complete and verifiable.
    assert manager_for(paths, smoke_config).newest_valid() is not None
    assert Manifest.load(paths.replay).verify() == []


# ===========================================================================
# The violent case (T24)
# ===========================================================================

RUNNER = """
import sys
from pathlib import Path
from reversi.config import load_config
from reversi.obs.runmeta import RunPaths
from reversi.train.loop import run_training

config = load_config(Path("configs/smoke4x4.yaml"), overrides=[
    "selfplay.games_per_generation=6",
    "train.steps_per_generation=2",
    "train.batch_size=8",
    "mcts.n_simulations=4",
    "net.n_blocks=1",
    "net.channels=8",
])
paths = RunPaths(run_id="resume-test", root=Path(sys.argv[1]))
run_training(config, paths, generations=8)
"""


def _tiny_config() -> Config:
    from reversi.config import load_config

    return load_config(
        Path("configs/smoke4x4.yaml"),
        overrides=[
            "selfplay.games_per_generation=6",
            "train.steps_per_generation=2",
            "train.batch_size=8",
            "mcts.n_simulations=4",
            "net.n_blocks=1",
            "net.channels=8",
        ],
    )


@pytest.mark.slow
def test_a_killed_run_resumes_without_losing_more_than_one_generation(
    tmp_path: Path,
) -> None:
    """Really kill a real process, then carry on.

    The kill is a hard terminate with no cleanup -- the same thing a reboot or an
    OOM kill does. Whatever state that leaves on disk is what resume has to cope
    with, including a checkpoint that was being written at the moment of death.
    """
    root = tmp_path / "run"
    script = tmp_path / "runner.py"
    script.write_text(textwrap.dedent(RUNNER), encoding="utf-8")

    process = subprocess.Popen(
        [sys.executable, str(script), str(root)],
        cwd=Path.cwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait until it has genuinely produced something, then kill it mid-flight.
    checkpoints = root / "checkpoints"
    deadline = time.monotonic() + 180
    try:
        while time.monotonic() < deadline:
            if list(checkpoints.glob("gen_*.json")):
                break
            if process.poll() is not None:
                pytest.fail(f"the run exited early:\n{process.communicate()[0]}")
            time.sleep(0.2)
        else:
            pytest.fail("no checkpoint appeared within the timeout")
        time.sleep(0.5)  # let it get into the middle of the next generation
    finally:
        process.kill()
        process.wait(timeout=30)

    config = _tiny_config()
    paths = RunPaths(run_id="resume-test", root=root)
    manager = manager_for(paths, config)

    survived = manager.newest_valid()
    assert survived is not None, "nothing usable survived the kill"
    reached = survived.meta.generation

    # Guard against a vacuous pass: if the run had already finished all eight
    # generations before the kill landed, there would be nothing to resume and
    # every assertion below would hold trivially.
    assert 1 <= reached < 8, (
        f"the run reached generation {reached} before being killed; the test needs "
        "it to be killed partway through to prove anything"
    )

    # ---- now resume ------------------------------------------------
    reports = run_training(config, paths, generations=8)

    assert reports, "the resumed run did no work"
    assert reports[0].generation == reached + 1, (
        f"resumed at generation {reports[0].generation} after reaching {reached}; "
        "at most one generation may be lost"
    )
    assert reports[-1].generation == 8, "the resumed run should finish the job"

    # ---- and everything it left behind is intact --------------------
    assert Manifest.load(paths.replay).verify() == [], "a replay shard was corrupted"
    for generation in manager.generations():
        manager.verify(paths.checkpoints / checkpoint_name(generation))


@pytest.mark.slow
def test_the_step_counter_does_not_restart_across_a_resume(
    tmp_path: Path,
) -> None:
    """The counter the learning-rate schedule reads.

    If it reset on every resume, a run split across three nights would repeat the
    warmup three times and never reach the rate it was configured with -- while
    looking perfectly healthy in the logs.
    """
    config = _tiny_config()
    paths = RunPaths(run_id="resume-test", root=tmp_path / "run")
    manager = manager_for(paths, config)

    run_training(config, paths, generations=2)
    after_two = manager.newest_valid()
    assert after_two is not None

    run_training(config, paths, generations=4)
    after_four = manager.newest_valid()
    assert after_four is not None

    per_generation = config.train.steps_per_generation
    assert after_two.meta.global_step == 2 * per_generation
    assert after_four.meta.global_step == 4 * per_generation, (
        "steps must accumulate across the restart, not start over"
    )


@pytest.mark.slow
def test_a_resumed_run_keeps_the_lineage_chain(tmp_path: Path) -> None:
    """Following ``parent`` backwards reconstructs the run across restarts.

    Generation numbers alone would not: two different attempts can both produce a
    "generation 3", and only the chain says which weights came from which.
    """
    config = _tiny_config()
    paths = RunPaths(run_id="resume-test", root=tmp_path / "run")
    manager = manager_for(paths, config)

    run_training(config, paths, generations=2)
    run_training(config, paths, generations=4)

    third = manager.read_meta(paths.checkpoints / checkpoint_name(3))
    fourth = manager.read_meta(paths.checkpoints / checkpoint_name(4))

    assert third.parent == checkpoint_name(2), "the first generation after a resume"
    assert fourth.parent == checkpoint_name(3)


def test_resuming_into_a_different_architecture_is_refused(
    smoke_config: Config, paths: RunPaths
) -> None:
    """Weights from a different network are not weights, they are noise.

    torch would reject an outright shape mismatch, but the message is unhelpful,
    and a change that happened to preserve shapes would not be caught at all.
    """
    from reversi.errors import CheckpointError

    run_training(smoke_config, paths, generations=1)

    wider = smoke_config.model_copy(
        update={"net": smoke_config.net.model_copy(update={"channels": 32})}
    )
    with pytest.raises(CheckpointError, match="different networks"):
        run_training(wider, paths, generations=2)
