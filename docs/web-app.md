# The web app

How to run it, what it is, and why it needs no machine of ours to think for it.

---

## "There is no server" — what that actually means

This is worth being precise about, because it is shorthand for something
specific rather than a claim that files appear out of nowhere.

**There is no *application* server.** Nothing of ours computes anything when you
play. No Python process runs a tree search, no session is stored, no request
carries your game anywhere. The rules, the search and the neural network all run
inside your browser tab.

**There is still a *file* server**, because something has to hand the browser the
HTML, the JavaScript, and the 1.8 MB network. But that job is the same job as
serving a photograph: it reads bytes off a disk and sends them. It does not care
what the game is, cannot be made slow by a hard position, and cannot run out of
capacity in any way that a hundred simultaneous players would notice. Once the
files arrive, the server is finished — you could unplug it mid-game and nothing
would change.

Here is the difference in one table:

| | The FastAPI version (still in the repo) | The site |
|---|---|---|
| Who runs the search | our machine | your machine |
| Cost per move | ~1 second of one CPU core | nothing to us |
| Concurrent players | 2, then `429 Too Many Requests` | no limit |
| If it has been idle | free tiers sleep, 30–50 s to wake | static files do not sleep |
| What a move sends | your whole position, over HTTP | nothing |

The FastAPI server has not been deleted, and §"Two engines" below says why.

**A consequence worth knowing:** you cannot double-click `index.html` and have it
work. Opening a file directly gives the page a `file://` origin, where JavaScript
modules, web workers and WebAssembly are all restricted. It has to be served over
HTTP. Every command below does that for you.

---

## Running it

### Once, to get the networks

The trained networks are not in git — model weights never are in this repo, and
`.onnx` is not an exception to that. Export them from a training run:

```bash
# a play-only copy of the checkpoint (no optimiser state), then the browser's copy
uv run reversi export runs/<run-id>/checkpoints/gen_00060.pt models/gen60.pt
uv run reversi export-onnx models/gen60.pt web/public/models/reversi-8x8-gen60.onnx
```

Repeat for generations 5, 20 and 40 if you want the full ladder; the
site offers whichever files are present and named in
`web/src/games/reversi/engine/models.json`.

`export-onnx` checks its own output. It runs the PyTorch model and the exported
one on the same random inputs and **deletes the file rather than keeping it** if
they disagree by more than float32 rounding. An export that loads but computes
something slightly different would not fail anywhere — the browser would simply
play an agent that none of the measurements in this repository describe.

### Every time

```bash
cd web
npm ci          # once, after cloning
npm run dev     # http://localhost:4173 — rebuilds when you save
```

`npm run dev` runs `vite build --watch` alongside `vite preview`, rather than the
Vite dev server. **It builds first** -- about two seconds -- then watches. Save a
file and it rebuilds in roughly half a second; refresh to see it. There is no hot
module replacement.

It does not type check on the way, deliberately: your editor already reports type
errors as you type, and a check standing between you and a running page is not a
trade worth making. `npm run build` does check, and so does CI, so nothing
untyped reaches a deploy. Run `npm run typecheck` yourself any time.

**Why not the normal dev server.** `vite` in dev transforms every module it
serves, and `onnxruntime-web` loads its own worker module with a dynamic
`import()`. Vite rewrites that import, appends `?import`, and then tries to
transform a prebuilt runtime file as if it were application source. The result
is that `InferenceSession.create` never settles: the page sits at "Loading the
agent…" forever with an empty console and no error to catch.

A build served statically has no transform pipeline, so none of it happens.
That is why `dev` is wired this way, and it has a useful side effect — what you
develop against is byte-for-byte what gets deployed.

`npm run dev:vite` still starts the real dev server if you want hot reloading
for pure layout or styling work. **The board will not work under it.** The
engine cannot load, so expect "Loading the agent…" and nothing else.

This is a known limitation rather than a solved problem. Fixing it properly
means either getting Vite to leave `/ort/` alone in dev entirely, or having the
runtime load its worker some other way.

### The tests

```bash
npm test                # the TypeScript engine against fixtures from the Python one
npx playwright test     # a whole game, in a real browser, against the built site
```

### Measuring the agent on a given device

Open `/bench/` — `http://localhost:4173/bench/`, or the same path on the
deployed site. It reports how long one forward pass takes and how long a move
takes at each simulation budget, on whatever machine loaded it. That page exists
because the decision to run the agent in the browser rests on numbers that
differ by more than tenfold between a laptop and a phone, and those are not
numbers to take on trust from one machine.

---

## Deploying it

**Cloudflare Pages builds this repository directly.** Connect the repo once in
the Cloudflare dashboard and every push to `main` deploys; every pull request
gets its own preview URL, which is genuinely useful for reviewing a change to the
board before it is merged.

### Build settings

| Setting | Value |
|---|---|
| Framework preset | None |
| Root directory | `web` |
| Build command | `npm run build:deploy` |
| Build output directory | `dist` |
| Environment variable | `NODE_VERSION` = `22` |

`build:deploy` is `build` with one step in front of it: fetching the trained
networks. That step exists because **the `.onnx` files are not in git.** They are
model weights, and the rule that keeps checkpoints out applies to them too — so a
build that clones the repository has the entire site except the thing that plays.
Without it the page renders a board and sits at "Loading the agent…" forever.

`npm run build` is untouched, so a local build never reaches for the network and
uses whatever you exported yourself.

### The models come from a Release

```bash
gh release create models-v1   web/public/models/*.onnx web/public/models/*.json   --title "Trained networks v1"   --notes "Generations 5, 20, 40 and 60 as ONNX."
```

About 7 MB. `scripts/fetch-models.mjs` reads the filenames from
`src/games/reversi/engine/models.json` rather than guessing them, downloads each one, and
**checks it against the SHA-256 recorded when it was exported** — before writing
it to disk, so a damaged download never lands where a later build would find it
and assume it was fine. A truncated file would otherwise be a valid-looking
`.onnx` that fails in the browser with no clue why.

The repository is public, so the build needs no credentials to fetch them. Set
`MODELS_TAG` to pin a build to an older set of weights, which is what you would
want to roll a deploy back to match a particular checkpoint.

### If the release does not exist yet

The build fails with the command that creates it. That is deliberate: a deploy
that quietly produced a site with no agent would be worse than one that stops.

### Why Cloudflare Pages and not GitHub Pages

GitHub Pages would have been simpler — it deploys from the repository already.
It was not chosen for one concrete reason: **it cannot set HTTP response
headers**, and this site needs two of them.

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Those two headers are what allow a page to use `SharedArrayBuffer`, and hence
multi-threaded WebAssembly. Without them the search runs on one thread and takes
**twice as long** — measured, see below. Cloudflare Pages reads them from
`web/public/_headers`.

The second reason is an option rather than a requirement: Cloudflare Workers and
Durable Objects live on the same platform, so if these games ever need to be
played between two people — the one feature that genuinely cannot be done
without a server — that is an addition rather than a migration.

---

## How fast it is, and why the top levels are timed

Measured in Chrome on a 20-core laptop, against the real generation-60 network:

| Simulations | 1 thread | 4 threads | Level |
|---|---:|---:|---|
| 16 | 127 ms | **55 ms** | Casual |
| 64 | 450 ms | **221 ms** | Club |
| 256 | 1,889 ms | **898 ms** | Strong |
| 800 | 5,503 ms | **2,789 ms** | Max |

Two things came out of taking these numbers rather than assuming them.

**Batching buys nothing here.** Throughput is flat from batch 1 to batch 32 —
146 to 170 positions per second. That means the cost is arithmetic rather than
the overhead of calling into WebAssembly, so collecting several search leaves and
scoring them together, which is exactly what makes self-play fast on the GPU,
would gain nothing at all in a browser. More threads was the only lever left.

**A count is not a budget.** A fixed number of simulations is a fixed amount of
*work*, and what a player experiences is *waiting* — and those differ by more
than tenfold between this laptop and a phone. So the two deep levels are capped
by a clock: `Strong` searches for up to 1.2 seconds and `Max` for up to 2.0,
spending however many simulations fit. On the machine above, `Max` reaches about
560 of its 800.

The interface reports the number it actually reached, and the level is labelled
"up to 2.0 s per move" rather than "800 simulations", because on most devices it
will not get there and saying otherwise would be the interface overstating how
hard the agent thought.

### There is also a floor

At the quicker levels a search finishes in under 300 ms, which is faster than a
person can follow. You click, your discs begin turning over, and before that
finishes the agent has moved and turned some of them back -- so the move you just
made is never actually visible and the board appears to change on its own.

So the agent will not answer sooner than 650 ms after your move, and a disc takes
300 ms to turn. The wait costs nothing: the search has already produced its
answer and this only delays showing it. Levels that genuinely take longer are
unaffected, so the pause exists where it is needed and nowhere else.

---

## What is in `web/`

```
web/
  index.html           the hub: one card per game, at /
  reversi/index.html   the Reversi page, at /reversi/
  bench/index.html     how fast is this device, at /bench/
  src/hub/             the front page and the list of games it shows
  src/shared/ui/       the frame every page shares
  src/games/reversi/
    engine/            the agent — no React anywhere in here
      bitboard.ts      a 64-square board in two 32-bit halves
      rules.ts         a port of reversi.game
      mcts.ts          a port of reversi.search (PUCT, FPU, time budget)
      features.ts      position → the three input planes
      onnx.ts          the network, via onnxruntime-web
      worker.ts        all of the above, off the main thread
      local.ts         the worker, behind a promise-shaped interface
      remote.ts        the FastAPI server, behind the same interface
      levels.ts        the four difficulty levels and the value guardrail
      hashStub.ts      a reproducible stand-in for the network, for tests
      models.json      the opponent list — generated, see below
      __fixtures__/    what the Python engine says the answers are
    state/             the game as a reducer over immutable positions
    ui/                React components: the board, the controls, the screen
  tests/               the engine against the fixtures, and the hub's registry
  e2e/                 a whole game in a real browser, and the hub
```

Each page is its own HTML file and there is no client-side router. A second game
is a new directory under `src/games/` and a new entry in `src/hub/registry.ts`;
nothing in the Reversi directory changes. Separate pages also keep Cloudflare's
`not_found_handling` on `"404-page"`: every URL maps to a file, so a missing
`.onnx` is a 404 rather than a page of HTML handed to the model loader.

### The search runs in a worker

A search is hundreds of network calls. On the page's main thread that would
freeze the tab for the whole of it — no hover, no scroll, and a "stop thinking"
button that could not be clicked. In a worker the board stays live, which is
also what makes cancelling a search a real action rather than a decoration.

### Two engines, one interface

The UI never learns where its opponent lives. It asks an `Engine` for a move:

```ts
interface Engine {
  legalMoves(pos: Position): Action[];
  apply(pos: Position, a: Action): Position;
  think(pos: Position, level: Level, signal: AbortSignal): Promise<Thought>;
}
```

`LocalEngine` is the worker. `RemoteEngine` is the FastAPI service in
`src/reversi/api/`, and keeping it costs about forty lines and one already-tested
server. It earns that three ways: a genuine fallback on a device where
WebAssembly is blocked, a way to compare the browser's answers against the
reference implementation during development, and the connection point if
two-player games are ever added.

This mirrors the `Evaluator` protocol on the Python side, which is what lets the
same search run against a real network or a stub without knowing the difference.

### The opponent list is generated, not typed

`src/reversi/web/manifest.py` reads the cross-generation tournament report and
writes `web/src/games/reversi/engine/models.json`:

```bash
uv run python -c "
from pathlib import Path
from reversi.web.manifest import build_manifest, write_manifest
m = build_manifest(Path('runs/<run-id>/arena/crossgen.json'))
write_manifest(Path('web/src/games/reversi/engine/models.json'), m)
"
```

The repository's rule is that no claim about strength is typed by hand, and the
way to keep a rule like that is to make breaking it impossible. If a rating
changes, everything derived from it changes with it; a generation that was never
rated cannot appear in the list at all.

### The ladder is derived from the ratings

A player picks **Level 1** to **Level 6**, not "Generation 20". The mapping lives
in `web/src/games/reversi/ladder.ts`, and it is computed rather than written:
the rungs are the opponents the app can play, sorted by the rating each measured
in the cross-generation round robin, numbered from the weakest.

| Level | Word | Plays as | Elo |
|---|---|---|---:|
| 1 | Beginner | Random | 0 |
| 2 | Easy | Greedy | 313 |
| 3 | Fair | Generation 5 | 547 |
| 4 | Tough | Generation 20 | 758 |
| 5 | Expert | Generation 40 | 855 |
| 6 | Expert | Generation 60 | 877 |

Three consequences worth stating.

**Adding a level is a data change.** Export a checkpoint, rate it in the
cross-generation tournament, regenerate the manifest, publish the `.onnx` in the
models release. A rung appears in the right place and every label renumbers
itself. No TypeScript changes. This is how `gen30` or `gen10` would be added if
the ladder ever wants finer steps near the middle, where the gap from level 3 to
level 4 is over 200 Elo.

**Levels 5 and 6 share a word on purpose.** Generations 40 and 60 are 22 rating
points apart with heavily overlapping intervals. Calling one "strong" and the
other "expert" would invent a difference the tournament did not find.

**One scale, never two.** Every number the ladder orders by comes from the
cross-generation table. The difficulty calibration below is a separate tournament
with its own fit, so its ratings are only ever shown beside the thinking time
they belong to — mixing the two would sort the ladder by numbers that were never
compared.

### Where the technical description went

Nothing was removed; it moved one click. The panel shows a level number and a
word, and a `<details>` beside it — "About level 4" — holds which checkpoint it
is, what it rates, the 95% interval, and what the current thinking time buys. The
launcher says "Six levels, beginner to expert" and keeps the ratings inside "How
the AI learned".

The reasoning: "Generation 20" means nothing before you know what a generation
is, and "+758 Elo" means nothing before you know what the scale is anchored to.
Both facts are worth reading once and neither is worth reading while it is your
move. A player who never opens the disclosure loses nothing; a reader who does
gets the whole claim including its error bar.

The count on the launcher is read from the ladder rather than from the manifest,
because the rating report holds two search baselines the app does not offer as
opponents. Counting report rows advertised eight levels and showed six.

---

## Reviewing a game is a cursor, not a copy

A board game you cannot look back at is a board game you cannot learn from. The
question after losing is never what the score was; it is where it went wrong, and
answering that needs the position from twelve moves ago on the screen.

The reducer already kept every position -- one immutable `Turn` per move, which is
what makes a take-back a matter of dropping the last entry. Reviewing adds one
field beside that history:

```ts
readonly viewing: number | null;   // an index into history, or null for live
```

**Two positions, and they are not the same one.** The *live* position drives the
engine, decides whose turn it is and ends the game. The *viewed* position is what
the board draws. They differ only while somebody is reviewing.

That separation is the whole feature. The screen restarts the agent's search
whenever the position, the opponent or the budget changes, and it identifies a
position by the history length. If reviewing worked by rewinding the game, then
scrubbing back while the AI was thinking would silently abort the move the player
was waiting for -- and the symptom is a board that never answers. Because
`viewing` is a second cursor over the same history, none of the things the search
keys on move, so the search finishes and the move lands. There is an end-to-end
test that does exactly this: start a search at the slowest thinking time, scrub
back mid-search, and assert the move still arrives.

**Anything that changes the game clears the cursor.** A move, a take-back, a new
game, a side swap. Reviewing a position that a new move has just invalidated is a
state with nothing sensible to draw, and a player who makes a move must see it
happen.

That includes the AI's reply arriving while somebody is mid-review, and the
decision there is deliberate. Staying put would be gentler -- the player chose to
look away -- but only the live position is interactive, so a board left in review
is a board that will not accept a move for a reason the player cannot see.
Snapping forward keeps the game playable, and the search still completes either
way, which is the part that would have been a bug.

**A stale index is clamped rather than trusted.** `undo` shortens the history, so
an index that survived would reach the board as `undefined` and render an empty
board instead of a position. `viewedPly` clamps; the reducer also clears on undo,
so it is belt and braces on the one failure that looks like a rendering bug
rather than a state bug.

**Landing on the last position counts as going live.** Otherwise scrubbing to the
end leaves the board inert on a position that happens to be the current one, with
no obvious way back.

The control is a native `<input type="range">`, restyled. Dragging, clicking the
track, arrow keys, Home and End, and the screen-reader announcement all come free,
and none of them would from two divs and a pointer handler. It is absent until
there is a move to review, because an inert slider on the opening position is a
promise the interface has not earned.

While reviewing, the turn indicator says so. It has to: the board is not showing
the game any more, and "Your turn" over a position from twelve moves ago is the
interface lying about what the player is looking at. The win-probability meter
stays visible during a review of a finished game, which is the one time it is
genuinely interesting -- scrubbing shows where the agent thought the game turned.

## The rules are written three times now

Twice in Python — an obvious list-of-lists version written as the specification,
and the fast bitboard version the agent was trained against — compared move by
move over 20,000 random games nightly and then frozen. The TypeScript engine is
the third, and it is held to the same standard.

**The expectations are generated, never hand-written.** A test written from
memory encodes what somebody believed the rules were; a generated one encodes
what the frozen engine actually does, which is what the agent was trained
against.

```bash
uv run reversi export-fixtures web/src/games/reversi/engine/__fixtures__ --onnx models/gen60.onnx
```

| Fixture | Contents | What it catches |
|---|---|---|
| `rules` | 1,000 positions → legal moves, the exact discs each move flips, terminal, score | edge wrap-around, a missed direction, the pass rule |
| `encoding` | 200 positions → the three input planes as bitmasks | a transposed board, the wrong player's point of view |
| `network` | 64 positions → the exported network's own outputs | a broken export, wrong input scaling |
| `search` | 40 positions → visit counts, deterministic evaluator, no noise | the value-sign inversion, PUCT arithmetic drift |
| `games` | 20 whole games, position after every move | anything the per-position fixtures let through |

CI regenerates all but `network` — which needs a trained model — and fails on
any difference. Without that check, someone could change the Python rules and
the browser would quietly stop being cross-checked against anything.

### It caught a real bug on its first run

Building the board mask computed `(1 << 32) - 1` for a full 32-bit half. But
JavaScript takes shift counts **modulo 32**, so `1 << 32` is `1`, not
2³², and the expression yields `0` instead of all ones.

The engine kept working. It simply stopped believing in the bottom four rows of
the board, and offered one legal opening move instead of four. Nothing crashed,
and in a game it would have looked like a weak agent rather than a broken port.

### The search agrees exactly

All 40 search positions match the Python reference **count for count**, which
was not the expectation. The two runtimes use different implementations of
`exp`, each allowed to differ by a unit in the last place, and one flipped
comparison early in a tree changes every count after it.

It works because the port rounds the evaluator's outputs to float32 with
`Math.fround` at precisely the points where Python stores them in a float32
array. Without that, the priors diverge in their last digits from the very first
node.

The test asserts 90% exact rather than 100%, because another browser engine is
entitled to land on the other side of one comparison. A drop below that is a
regression, not rounding.

---

## Two knobs, on purpose

The panel has two selects, and they are the two halves of the method.

**Level** picks which opponent plays: a real checkpoint from the self-play run,
or one of the two rule-only baselines. That sets how good the agent's intuition
is. The ratings come from a round robin of 210 games per entrant, fitted with
Bradley–Terry and anchored so random play is 0.

**Thinking time** sets how much search runs on top of that intuition before it
moves.

Keeping them separate is deliberate. A player can face the final agent thinking
briefly, or an early agent thinking hard, and the difference between those two
is the whole idea behind the method: the network supplies a fast opinion, and the
search improves on it. A single difficulty slider would hide that, and the levels
would stop being measurable things.

| Level | Plays as | Elo | 95% interval | Runs |
|---|---|---:|---|---|
| 6 | Generation 60 | 877 | 774 – 1028 | the network |
| 5 | Generation 40 | 855 | 747 – 1018 | the network |
| 4 | Generation 20 | 758 | 659 – 898 | the network |
| 3 | Generation 5 | 547 | 467 – 686 | the network |
| — | *Minimax, depth 4* | *523* | *434 – 653* | *rated, not offered as a level* |
| 2 | **Greedy** | **313** | **220 – 468** | **the rules alone** |
| 1 | **Random** | **0** | — | **the rules alone** |

### Why levels 1 and 2 are not the network

The network is strong even at its earliest checkpoint. Generation 5 rates +547,
which is *above* the depth-4 alpha-beta search it was measured against, and no
simulation budget changes that: the strength is in the network's own opinion and
the search only sharpens it. Turning `Casual` down further makes the agent slower
to decide, not weaker.

So the bottom of the ladder is Random and Greedy — the same baselines every
rating in this project is measured against, rated in the same round robin as
every generation. Without them the easiest level on offer was already stronger
than a classical engine, and a new player had nothing to beat.

They also cost nothing. Neither needs the 1.8 MB network, so picking one is
instant and a visitor who only plays level 2 downloads no model at all — which is
why level 2 is where the page starts.

Neither reports a win probability or a simulation count, and the interface hides
both rather than inventing them: one picks uniformly at random and the other
counts discs. Neither holds an opinion about who is winning. The thinking-time
select is hidden for these two as well, because there is no search to budget.

### The value guardrail

At the easier levels the agent samples among reasonable moves rather than always
taking the best one, or it would be the same opponent every game. The guardrail
throws away candidates whose value is worse than the best by more than a
threshold, **and then** samples among what survives.

That order matters. Sampling first and checking afterwards would let a genuine
blunder through whenever it happened to be drawn, which is exactly what the
guardrail exists to prevent. It is what makes an easy opponent *weak* rather than
*stupid* — it plays a second-best move instead of giving away a corner.

### The thinking times are calibrated

The four thinking times were rated against each other and against the frozen
baselines, all on one checkpoint, in a single Bradley–Terry fit anchored at random
play = 0. Using one network is the point: if the rungs separate, the separation
comes from the search rather than from four different models.

| Thinking time | Simulations | Elo | 95% interval | Gap below |
|---|---:|---:|---|---:|
| Casual | 16 | 431 | 400 – 465 | — |
| Club | 64 | 623 | 588 – 664 | +192 |
| Strong | 256 | 891 | 845 – 943 | +267 |
| Max | 800 | 1053 | 1001 – 1115 | +163 |

Criterion S15 — strictly monotonic ratings, non-overlapping intervals, gaps of at
least 80 Elo — is met with room to spare, and the guardrail held over 500 of
Casual's moves at a worst drop of 0.326 against its 0.35 limit. Full setup,
including the run where it first failed, is in `docs/experiments.md`.

These numbers sit on their own scale. They are shown in the app only inside the
note beside a level, never mixed into the ladder's ordering.

---

## Troubleshooting

**Blank page, nothing in the console.** You probably opened `index.html` from
disk. Serve it over HTTP: `npm run dev` or `npm run preview`.

**End-to-end tests fail on code you can see is correct.** A `vite preview` left
listening on port 4173 is reused instead of started, and `reuseExistingServer`
skips the build with it -- so the tests run against whatever `dist/` held when
that server came up. The symptom is a selector that cannot find an element you
are looking at in your own browser. Kill the listener and run again:

```powershell
Get-NetTCPConnection -LocalPort 4173 -State Listen | ForEach-Object {
    Stop-Process -Id $_.OwningProcess -Force
}
```

This has now caused two rounds of confusion, both times looking like a bug in the
component under test.

**"Loading the agent…" forever.** The `.onnx` file is missing. Check
`web/public/models/` against the URLs in `web/src/games/reversi/engine/models.json`. If it is
present, open the network tab: a 404 there is the answer.

**It hangs at "Loading the agent…" with an empty console.** Two causes, and they
look identical.

*You are on `npm run dev:vite`.* The Vite dev server cannot serve the ONNX
runtime — see "Every time" above. Use `npm run dev`.

*Otherwise, it is the threading path.* The runtime starts its worker threads from
a sibling `.mjs`, and if the bundler has renamed or moved that file the worker
cannot find it and nothing reports the problem. `src/games/reversi/engine/onnx.ts` sets
`wasmPaths` to a fixed directory and `scripts/stage-runtime.mjs` fills it; if you
change either, change both.

**Every move takes twice as long as the table above.** Cross-origin isolation is
off, so the runtime fell back to one thread. Check `crossOriginIsolated` in the
console — if it is `false`, the two headers are not arriving. `vite dev` and
`vite preview` both send them; a different static server may not.

---

## What is deliberately not here

| Not built | Why | What would change it |
|---|---|---|
| Human vs human | the only feature that genuinely needs a server, and a different product from "play the thing I trained" | wanting it — then Cloudflare Durable Objects, on the host already chosen |
| Accounts, leaderboards, saved games | a database, a privacy story and moderation, none of which demonstrate anything about the agent | — |
| WebGPU | for a network this small, per-operation dispatch overhead usually costs more than the arithmetic it saves; the WebGPU build is also twice the download | a measurement showing otherwise on real devices |
| Server-side rendering | a board game behind a click needs no search-engine story and no origin process | — |
