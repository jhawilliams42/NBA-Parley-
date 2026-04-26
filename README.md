# NBA Prop Engine

This repo contains the modular `v15.1` NBA prop engine plus an end-to-end
orchestrator exposed as `nba_prop_engine.build_nba_prop_portfolio`.

## What is runnable now

- Phase 1 through Phase 5 component modules remain independently testable.
- `nba_prop_engine.engine` stitches those phases into a single portfolio build.
- `python -m nba_prop_engine.cli <input.json>` runs the full engine from a JSON payload.

## Input contract

The CLI and Python API expect a payload with:

- `run_context`
- `snapshot_bundle`
- `player_game_objects`
- optional `valuation_books_map`

`valuation_books_map` may contain either already-processed Phase 2 valuation
records or raw per-book American odds:

```json
{
  "P001": [
    {"book_name": "DraftKings", "odds_over_american": -125, "odds_under_american": 105},
    {"book_name": "BetMGM", "odds_over_american": -125, "odds_under_american": 105}
  ]
}
```

## API keys

The current codebase does not load API keys, tokens, or `.env` files. The
engine runs on caller-supplied snapshot data only.
