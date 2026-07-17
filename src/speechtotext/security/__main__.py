from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
import sys

from speechtotext.security.artifacts import PrivateArtifactStore


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.exit(2, "ERROR code=invalid_arguments\n")


def build_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(prog="python -m speechtotext.security")
    commands = parser.add_subparsers(
        dest="command", required=True, parser_class=SafeArgumentParser
    )
    promote = commands.add_parser("promote", add_help=False)
    promote.add_argument("--source", type=Path, required=True)
    promote.add_argument("--name", required=True)
    promote.add_argument("--expected-sha256", required=True)
    promote.add_argument("--max-bytes", type=int, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    store_factory: Callable[[], PrivateArtifactStore] = (
        PrivateArtifactStore.current_user
    ),
) -> int:
    args = build_parser().parse_args(argv)
    try:
        store_factory().promote_from_path(
            args.source,
            args.name,
            expected_sha256=args.expected_sha256,
            max_bytes=args.max_bytes,
        )
    except Exception:
        print("ERROR code=artifact_promotion_failed", file=sys.stderr)
        return 1
    print("OK artifact_promoted=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
