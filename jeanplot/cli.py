"""Minimal CLI entrypoints for package scripts."""

from __future__ import annotations

import argparse

from jeanplot import load_default_theme


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jeanplot")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("theme-check", help="load the default theme and report status")

    args = parser.parse_args(argv)
    if args.command == "theme-check":
        load_default_theme(force=True)
        print("theme ok")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
