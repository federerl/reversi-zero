# Demo checklist

What to check before showing the web app, and what each item is there to prove.
This is criterion S17: the interface has to be usable, not merely functional, and
"usable" has to mean something you can fail.

Run against a production build, not the dev server:

```bash
cd web && npm run build && npx vite preview --port 8123 --strictPort
```

The dev server does not serve this app (Vite's transform pipeline chokes on
onnxruntime-web's prebuilt worker), so checking against it would be checking
something nobody ships. See `docs/web-app.md`.

## The board

- [ ] Eight equal columns and eight equal rows, at every window size. The board
      stays square; no row or column is thinner than its neighbours.
- [ ] Legal moves for the side to move are marked with dots.
- [ ] Playing a move flips the bracketed discs, and the flip is visible rather
      than instantaneous — you should be able to see what your move did.
- [ ] The agent's last move is highlighted, so you can tell what it just played
      without replaying the game in your head.
- [ ] Disc counts update for both sides after every move.

## Knowing what is going on

- [ ] The status line says whose turn it is.
- [ ] While the agent is searching, the board is locked and something says so.
      A board that silently ignores clicks reads as broken.
- [ ] After the agent moves, the panel reports the square, the simulation count,
      and the elapsed milliseconds.
- [ ] "Agent's estimate" shows its own view of who is winning, as a percentage
      with a bar, and the label says which side the percentage is for.

## Passing and ending

- [ ] With no legal move available, the interface says so and offers the pass
      explicitly rather than leaving a dead board.
- [ ] The agent passes on its own when it must, and says that it did.
- [ ] At the end, the winner and the final score are both shown.
- [ ] A game can end before the board is full — Reversi allows it — and the
      terminal panel handles that case the same way.

## The claims the interface makes

- [ ] Every opponent in the dropdown is labelled with a **measured** rating and
      its 95% interval, never an adjective. "Generation 60 — 877 Elo", not
      "Hard".
- [ ] Every thinking-time setting is labelled the same way.
- [ ] No entry appears that has not been rated. The manifest build fails rather
      than showing an unmeasured option, so if the list renders at all, every row
      in it was measured.
- [ ] The caption under the board explains where the numbers came from and admits
      that neighbouring generations overlap.

## Controls

- [ ] New game resets the board and the score.
- [ ] Swap sides keeps the opponent and difficulty, changes only colour.
- [ ] Take back is disabled at the start of a game and enabled once a move
      exists.
- [ ] "Show where the agent searched" toggles the visualisation and survives a
      move.

## Keyboard and screen readers

- [ ] Every square is reachable by keyboard and playable with Enter.
- [ ] Focus is visible — you can always tell which square you are on.
- [ ] Each square's accessible name gives its coordinate, its contents, and
      whether it can be played: "d3, empty — you can play here".
- [ ] Legality is not communicated by colour alone.

## Sizes

- [ ] Usable at 375 px wide — the board stays square and the panel stacks below.
- [ ] Usable at 1440 px — the board does not stretch to absurd proportions.

## Failure paths

- [ ] With the model files missing, the page says what is missing rather than
      hanging on "Loading the agent…".
- [ ] On a browser without the threaded WASM path, it falls back to
      single-threaded and still plays, more slowly.
- [ ] Loading takes a few seconds the first time — the network is 1.76 MB and
      ONNX Runtime has to start — and the interface says it is loading rather
      than appearing broken.

## Before showing it to anyone

- [ ] Load the page fresh, with an empty cache, and time it. The first load is
      the one your audience sees.
- [ ] Play one full game to the end at Casual and one at Max, so nothing in the
      terminal path is a surprise.
- [ ] Open `/bench/` once — it measures the visitor's own machine, and it is the
      answer to "how fast is it really".
