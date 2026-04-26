"""
Simple JSON CLI for the end-to-end NBA prop portfolio builder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .engine import build_nba_prop_portfolio


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the NBA prop engine from a JSON payload.")
    parser.add_argument("input", help="Path to a JSON payload with run_context, snapshot_bundle, and player_game_objects.")
    parser.add_argument("-o", "--output", help="Optional output path for the result JSON.")
    parser.add_argument("--target-tickets", type=int, default=21, help="Target number of portfolio tickets.")
    parser.add_argument("--kelly-fraction", type=float, default=1 / 8, help="Fractional Kelly multiplier.")
    args = parser.parse_args(argv)

    payload = _load_json(args.input)
    result = build_nba_prop_portfolio(
        run_context=payload["run_context"],
        snapshot_bundle=payload["snapshot_bundle"],
        player_game_objects=payload["player_game_objects"],
        valuation_books_map=payload.get("valuation_books_map"),
        target_tickets=args.target_tickets,
        kelly_fraction=args.kelly_fraction,
    )
    output = json.dumps(result.to_dict(), indent=2, sort_keys=True)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
