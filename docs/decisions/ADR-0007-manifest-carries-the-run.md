# ADR-0007: A published model is identified by its run, not by its generation

**Status:** accepted
**Applies to:** `web/manifest.py`, `web/src/games/reversi/engine/models.json`, the models release
**Supersedes:** manifest version 1
**Contracts:** none directly; protects the rule that no strength claim is typed by hand

## The problem

The web app picks its opponents from a generated file. Version 1 of that file
described a model with two facts: which generation of the training run it was,
and what it rated. The URL it fetched followed from the first:

```
/models/reversi-8x8-gen60.onnx
```

That is complete and unambiguous while exactly one training run exists. It stops
being either as soon as there are two.

By September 2026 there were eight. Run 1 is a 6×64 network trained without an
ownership head; run 6 is 10×128; runs 5, 7 and 8 are 6×64 with the head, on
different seeds. Every one of them has a generation 60, and they are different
networks with different strengths. "Generation 60" names four things.

Three consequences, in increasing order of how badly they hurt.

**The file name collides.** Two runs' generation 60 both want to be
`reversi-8x8-gen60.onnx`. Release assets live in one flat namespace per release,
so there is no directory to separate them: whichever file is uploaded second
overwrites the first, or the upload is rejected and the older file silently
stays.

**A rating can end up on the wrong weights.** This is the failure that matters.
The manifest's job is to attach a measured number to a specific network. If the
URL resolves to a different network than the one that was rated, the app shows a
real rating from a real tournament next to weights that never played in it.
Nothing downstream can detect that. The page loads, the agent plays well, the
number is plausible, and the claim is false.

**A reader cannot tell what they are downloading.** A 6×64 network is 1.8 MB and
searches quickly; a 10×128 is roughly four times the arithmetic per simulation,
which changes what a thinking-time cap can reach. Version 1 recorded neither the
architecture nor a checksum, so the only way to find out was to fetch the file.

## The decision

**Version 2 identifies a model by its run, and refuses to write a manifest where
two models could be confused.**

Concretely, four changes:

1. **Each model entry records `run`, `arch` and `sha256`**, read from the export
   sidecar that `reversi export-onnx` writes beside each `.onnx`. They are not
   restated in the manifest generator, because the sidecar is where those facts
   are authoritative and a second copy is a second thing to get wrong.

2. **The file name carries a run key**: the run id's start date and time, which
   is what makes a run id unique in the first place. So generation 60 of run 1
   is `reversi-8x8-gen60-20260827-030939.onnx`. The full run id stays in the
   manifest entry — the name is shortened for readability, not truncated for
   storage.

3. **The builder checks the URLs it produced and refuses a collision.** This is
   the actual guarantee. A naming convention is a hope; a check is a promise. If
   two entries ever resolve to one URL — however the key is derived — the build
   fails with both run ids named, rather than writing a file that would serve one
   network under the other's rating.

4. **A model with no sidecar cannot be published at all.** No checksum means no
   way to verify the file the manifest points at, and an unverifiable file with a
   rating beside it is worse than an unrated one: it looks like evidence.

The manifest also gained `game` and `release`. The release tag is what makes the
URLs resolvable; the game is what lets a second game's manifest exist beside this
one in 1.1 without either having to guess.

## Why not the alternatives

**Put the run in a directory instead of the file name.**
`/models/<run>/reversi-8x8-gen60.onnx` reads better and was the first choice. It
does not survive the hosting: GitHub Release assets are a flat namespace, so the
asset name itself has to be unique. Serving the files from somewhere with
directories would mean paying for and maintaining that somewhere, which ADR-0005
decided against for the same reason it decided against a server.

**Hash the weights into the name.** `reversi-8x8-gen60-89dd1736.onnx` is unique
by construction and needs no rule about run ids. It is also unreadable: a person
looking at a release cannot tell which of two files is newer, or which run
either came from. The checksum is recorded in the manifest, where a machine can
use it, and the human-readable key is in the name, where a person can.

**Keep version 1 and rely on care at release time.** The release step is where
the mistake would happen, performed rarely, by hand, months apart. That is
exactly the situation the rest of this repository refuses to rely on care for —
the manifest is generated rather than typed for the same reason.

## What this does not do

**It does not renumber anything, and it publishes nothing.** Building a manifest
writes a JSON file the site reads at build time. Uploading the matching `.onnx`
files to a release is a separate, deliberate step, and the committed manifest
stays at version 1 until that step is taken. The two must move together: a
version 2 manifest names files that only exist once the release carrying them
does.

**It does not break version 1 consumers.** Version 2 adds keys and renames none,
which is why the web app needed no change. A test pins that, so a future version
that has to remove a key makes that decision visible rather than incidental.

**It does not put two runs on one scale.** The manifest can now *describe* models
from different runs; it cannot make their ratings comparable. Two tournaments
that share only an anchor agree on the anchor and nothing else. Listing run 1's
generation 60 beside run 5's requires them to have played each other in one round
robin, which is a measurement, not a file format.
