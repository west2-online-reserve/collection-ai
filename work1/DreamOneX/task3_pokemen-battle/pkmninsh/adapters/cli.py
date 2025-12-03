from __future__ import annotations

import argparse
import os
import sys
from typing import List

from pkmninsh.init import init_app
from pkmninsh.infra.i18n import translate as t
from pkmninsh.app.session import GameSession
from pkmninsh.app.player import LocalPlayer, Computer


def _split_plugin_paths(raw: str) -> list[str]:
    """
    Split plugin paths using system-appropriate path separator.

    Uses semicolon (;) on Windows, colon (:) on Unix-like systems.

    Args:
        raw: Raw path string with multiple directories

    Returns:
        List of non-empty directory paths
    """
    separator = ";" if os.name == "nt" else ":"
    return [path.strip() for path in raw.split(separator) if path.strip()]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments with localized help text.

    Language is detected first to properly render argparse messages.
    Supports language override via PKMNNISH_LANG environment variable.

    Args:
        argv: Command-line arguments (uses sys.argv if None)

    Returns:
        Parsed argument namespace
    """
    # Initialize app early to enable i18n for argparse help text
    env_lang = os.getenv("PKMNNISH_LANG")
    # Get plugin dirs from env for early initialization
    plugins_env = os.getenv("PKMNNISH_PLUGINS_DIRS", "plugins")
    plugin_dirs = _split_plugin_paths(plugins_env)
    init_app(lang=env_lang, plugin_dirs=plugin_dirs)

    parser = argparse.ArgumentParser(
        prog="pkmninsh",
        description=t("cli.description"),
        epilog=t("cli.epilog"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--plugins",
        default=os.getenv("PKMNNISH_PLUGINS_DIRS", "plugins"),
        help=t("cli.plugins_help"),
        metavar="PATHS",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=t("cli.seed_help"),
        metavar="INT",
    )

    parser.add_argument(
        "--lang",
        default=env_lang,
        help=t("cli.lang_help"),
        metavar="CODE",
    )

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """
    Main entry point for the CLI application.

    Orchestrates game setup, player selection, and battle loop.
    All user-facing text is localized via the i18n system.

    Args:
        argv: Command-line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        args = parse_args(argv)
    except SystemExit as e:
        # argparse calls sys.exit on --help or parse errors
        return e.code if isinstance(e.code, int) else 1

    # Re-initialize if user specified --lang flag or different --plugins
    # Parse plugin directories first
    plugin_dirs = _split_plugin_paths(args.plugins)
    plugins_env = os.getenv("PKMNNISH_PLUGINS_DIRS", "plugins")
    env_plugin_dirs = _split_plugin_paths(plugins_env)

    # Re-initialize if lang or plugins differ from env
    if args.lang or plugin_dirs != env_plugin_dirs:
        init_app(lang=args.lang, plugin_dirs=plugin_dirs)

    try:
        # Initialize game session
        print(t("boot.loading"))
        session = GameSession(plugin_dirs=plugin_dirs, seed=args.seed)
        print(t("boot.ready"))
        print()

        # Initialize players
        player = LocalPlayer()
        enemy = Computer()

        # Creature selection phase
        session.setup(player, enemy)
        snap = session.snapshot()
        print(
            t("game.selected", variables={"player": snap.player_name, "enemy": snap.enemy_name})
        )

        # Main battle loop
        while not session.is_over():
            # TODO: implement more detailed turn info display
            snap = session.snapshot()
            print()
            print(
                t(
                    "state.line",
                    variables={
                        "name": snap.player_name,
                        "hp": snap.player_hp,
                        "max": snap.player_max_hp,
                    },
                )
            )
            print(
                t(
                    "state.line",
                    variables={
                        "name": snap.enemy_name,
                        "hp": snap.enemy_hp,
                        "max": snap.enemy_max_hp,
                    },
                )
            )

            session.step(player, enemy)

            while log := session.pop_log():
                print(log)


        # Display battle result
        winner = session.winner()
        print()

        if winner == "player":
            print(t("game.you_win"))
        elif winner == "enemy":
            print(t("game.you_lose"))
        else:
            print(t("game.draw"))

        return 0

    except KeyboardInterrupt:
        print()
        print("Game interrupted by user")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        raise


if __name__ == "__main__":
    sys.exit(main())
