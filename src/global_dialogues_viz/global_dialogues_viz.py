from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

from kash.exec import kash_action
from kash.model import Item, Param
from kash.workspaces import current_ws

from global_dialogues_viz.data_cleanup import simplify_csv

log = logging.getLogger(__name__)

# Participants data.
# NB: keep the typo "particpants"! (not "participants").
CSV_URL = "https://huggingface.co/datasets/collective-intelligence-project/Global-AI-Dialogues/raw/main/Global%20AI%20Dialogues%20Data%20-%20September%202024/particpants.csv"
TARGET_COLUMNS = [
    "Participant Id",
    "How old are you?",
    "What is your gender?",
    "What religious group or faith do you most identify with?",
    "What country or region do you most identify with?",
    "What do you think your life might be like in 30 years? Alt: Imagine life 30 years from now. What's the biggest difference you notice in daily life compared to today? (English)",
]


@kash_action(
    params=(Param("max_rows", "Maximum number of rows to include in the visualization", type=int),)
)
def simplify_participant_csv(item: Item, max_rows: int = 0) -> Item:
    """
    Clean up/simplify the participant CSV file.
    """
    ws = current_ws()
    assert item.store_path

    simplified_data = item.derived_copy(title="participants_simple")
    target_path = ws.target_path_for(simplified_data)

    log.warning("Simplifying data to: %s", target_path)

    simplify_csv(ws.base_dir / item.store_path, target_path, TARGET_COLUMNS, max_rows)
    simplified_data.external_path = str(target_path)

    log.warning("Simplified data: %s", simplified_data)

    return simplified_data


def create_viz(ws_path: Path, data_url: str, style: str, max_rows: int = 0) -> str:
    """
    Use a kash workspace to clean up and visualize the participants CSV file.
    """
    from kash.exec import kash_runtime, prepare_action_input
    from kash.kits.experimental.actions.embed_table_rows import embed_table_rows
    from kash.kits.experimental.actions.show_embeddings_graph_view import show_embeddings_graph_view

    # Run all actions in the context of this workspace.
    with kash_runtime(ws_path) as runtime:
        # Show the user the workspace info.
        runtime.workspace.log_workspace_info()

        # Prepare the URL input and run transcription.
        log.warning("Fetching data: %s", data_url)

        input = prepare_action_input(data_url)
        orig_data = input.items[0]

        log.warning("Original data: %s", orig_data)

        simpler_data = simplify_participant_csv(orig_data, max_rows=max_rows)

        log.warning("Simplified data: %s", simpler_data)

        embedding_data = embed_table_rows(simpler_data)

        final = show_embeddings_graph_view(embedding_data, style=style)

        log.warning("Final: output file: %s", final.store_path)

        assert final.store_path
        return final.store_path


def get_log_level(args: argparse.Namespace) -> Literal["debug", "info", "warning", "error"]:
    if args.quiet:
        return "error"
    elif args.verbose:
        return "info"
    elif args.debug:
        return "debug"
    else:
        return "warning"


def add_general_flags(parser: argparse.ArgumentParser) -> None:
    """
    These are flags that should work anywhere (main parser and subparsers).
    """
    parser.add_argument(
        "--debug", action="store_true", help="enable debug logging (log level: debug)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="enable verbose logging (log level: info)"
    )

    parser.add_argument("--quiet", action="store_true", help="only log errors (log level: error)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Global AI Dialogues data")
    add_general_flags(parser)

    parser.add_argument(
        "--style",
        choices=["2d", "3d"],
        default="3d",
        help="Visualization style, '2d' or '3d'",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=500,
        help="Number of rows to use for visualization (0 = no limit)",
    )

    args = parser.parse_args()

    # Set up kash workspace.
    from kash.commands.base.show_command import show
    from kash.config.setup import kash_setup

    ws_root = Path(".")
    ws_path = ws_root / "workspace"

    kash_setup(kash_ws_root=ws_root, rich_logging=True, console_log_level=get_log_level(args))

    output_path = create_viz(ws_path, CSV_URL, args.style, args.rows)

    show(output_path)


if __name__ == "__main__":
    main()
