"""Build an untrained network in the shape the web app expects.

Continuous integration has no trained checkpoint to play against -- weights are
not in the repository -- but the browser tests still need *a* network at the URL
the manifest names, or the page cannot finish loading and every one of them
fails for a reason that has nothing to do with the change being tested.

So this writes a randomly initialised network of the right architecture. It
plays badly, which does not matter: these tests check that a move can be made,
that the board advances, that a game reaches an end. None of them assert
anything about strength, and none of them could -- an untrained agent has none.

Never run this to produce something a person will play against. The filename it
writes is deliberately the same one the real model uses, because the point is to
stand in for it, and that is exactly why it belongs in CI and nowhere else.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from reversi.ckpt.manager import CheckpointManager
from reversi.config import NetConfig
from reversi.nn.export import export_checkpoint
from reversi.nn.model import build
from reversi.nn.onnx import export_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="Directory to write the .onnx into")
    parser.add_argument("--board-size", type=int, default=8)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("web/src/engine/models.json"),
        help="Read the expected filenames from here, so the names always match.",
    )
    # Small on purpose. CI is not measuring inference speed, and a full-size
    # network would add a download and a compile to every browser test.
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--channels", type=int, default=16)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    names = [Path(entry["url"]).name for entry in manifest["models"]]

    args.destination.mkdir(parents=True, exist_ok=True)
    net = build(
        NetConfig(n_blocks=args.blocks, channels=args.channels, value_hidden=32),
        args.board_size,
        seed=0,
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manager = CheckpointManager(root / "ckpt", run_id="placeholder", config_sha256="")
        manager.save(model=net, generation=0, global_step=0)

        staged = root / "placeholder.pt"
        export_checkpoint(root / "ckpt" / "gen_00000.pt", staged)

        first = args.destination / names[0]
        meta = export_onnx(staged, first)
        print(f"  {names[0]}  {meta['bytes'] / 1e6:.2f} MB  (untrained placeholder)")

        # `export_onnx` guarantees the weights are inside the file, so copying
        # the .onnx copies the whole model. Asserting it here rather than
        # trusting it: a model whose weights lived in a sidecar would copy as a
        # file that still referred to the original by name, and would fail only
        # once a browser tried to load it.
        from reversi.nn.onnx import _has_external_data

        if _has_external_data(first):
            msg = f"{first.name} keeps its weights outside the file; it cannot be copied"
            raise SystemExit(msg)

        # Every opponent in the manifest points at the same file. A test that
        # switches generations is checking that switching works, not that the
        # generations differ.
        for name in names[1:]:
            (args.destination / name).write_bytes(first.read_bytes())
            (args.destination / name).with_suffix(".json").write_text(
                first.with_suffix(".json").read_text(encoding="utf-8"), encoding="utf-8"
            )
            print(f"  {name}  (copy)")


if __name__ == "__main__":
    main()
