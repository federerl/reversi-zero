"""Atomic writes (backlog T04; underpins criteria S8 and T23).

The failure being prevented: a job killed mid-write leaves a truncated file that
loads without error but contains garbage, silently poisoning a resume.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reversi.atomicio import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    atomic_write_with,
    sha256_file,
    sha256_text,
)


def test_write_text_and_json(tmp_path: Path) -> None:
    text_path = tmp_path / "a.txt"
    atomic_write_text(text_path, "hello\n")
    assert text_path.read_text(encoding="utf-8") == "hello\n"

    json_path = tmp_path / "b.json"
    atomic_write_json(json_path, {"b": 2, "a": 1})
    assert json_path.read_text(encoding="utf-8").index('"a"') < json_path.read_text(
        encoding="utf-8"
    ).index('"b"'), "keys are sorted, so hashes are stable"


def test_parent_directories_are_created(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "file.txt"
    atomic_write_text(target, "x")
    assert target.is_file()


def test_overwrite_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.bin"
    atomic_write_bytes(path, b"v1")
    atomic_write_bytes(path, b"v2")
    assert path.read_bytes() == b"v2"


def test_failed_write_leaves_the_previous_version_intact(tmp_path: Path) -> None:
    """This is the whole point: a crash mid-save must not destroy the last good file."""
    path = tmp_path / "latest.pt"
    atomic_write_text(path, "good")

    def exploding_writer(tmp: Path) -> None:
        tmp.write_text("partial", encoding="utf-8")
        msg = "simulated kill during save"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="simulated kill"):
        atomic_write_with(path, exploding_writer)

    assert path.read_text(encoding="utf-8") == "good"


def test_failed_write_leaves_no_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "latest.pt"

    def exploding_writer(tmp: Path) -> None:
        tmp.write_bytes(b"junk")
        raise OSError

    with pytest.raises(OSError, match=""):
        atomic_write_with(path, exploding_writer)

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_atomic_write_with_delivers_the_writers_output(tmp_path: Path) -> None:
    path = tmp_path / "shard.npz"
    atomic_write_with(path, lambda tmp: tmp.write_bytes(b"payload"))
    assert path.read_bytes() == b"payload"


def test_sha256_helpers_agree(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    atomic_write_text(path, "abc")
    assert sha256_file(path) == sha256_text("abc")


def test_sha256_file_detects_a_single_flipped_byte(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    atomic_write_bytes(path, b"\x00" * 1024)
    before = sha256_file(path)
    atomic_write_bytes(path, b"\x00" * 512 + b"\x01" + b"\x00" * 511)
    assert sha256_file(path) != before
