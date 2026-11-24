#!/usr/bin/env python3
"""Print the official IBM TTM-R3 route matrix without downloading weights."""

import argparse
import csv
import sys


def parse_ints(value: str) -> list[int]:
    """Parse a comma-separated integer list.

    Args:
        value: Comma-separated values.

    Returns:
        Parsed integers in input order.
    """
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    """Print the resolved TTM-R3 route matrix.

    Returns:
        None; rows are written to standard output.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="32,64,128,256,512")
    parser.add_argument("--horizons", default="1,6,12,24,48,96")
    parser.add_argument(
        "--prefer-known-mappings",
        choices=("true", "false"),
        default=None,
        help=(
            "Use IBM's legacy known mappings. Defaults to "
            "TTM_PREFER_KNOWN_MAPPINGS or false."
        ),
    )
    args = parser.parse_args()

    # Keep package-level help usable without the model-specific TTM environment.
    from models.ttm_r3_auto import (
        prefer_known_mappings_setting,
        resolve_ttm_r3_revision,
    )

    prefer_known_mappings = prefer_known_mappings_setting(
        args.prefer_known_mappings
    )

    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "context_len",
            "horizon_len",
            "prefer_known_mappings",
            "model_variant",
        ]
    )
    for context_len in parse_ints(args.contexts):
        for horizon_len in parse_ints(args.horizons):
            writer.writerow(
                [
                    context_len,
                    horizon_len,
                    str(prefer_known_mappings).lower(),
                    resolve_ttm_r3_revision(
                        context_len,
                        horizon_len,
                        prefer_known_mappings,
                    ),
                ]
            )


if __name__ == "__main__":
    main()
