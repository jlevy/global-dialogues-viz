from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from kash.exec import kash_action
from kash.model import Item, Param
from kash.utils.file_utils.csv_utils import sniff_csv_metadata
from kash.workspaces import current_ws

log = logging.getLogger(__name__)


def simplify_csv(
    input_path: Path,
    output_path: Path,
    target_columns: list[str],
    max_rows: int = 0,
    column_transformations: dict[str, Callable[[str], str]] | None = None,
) -> None:
    """
    Process the CSV file to clean up format and extract specific columns.
    """
    log.info("Processing CSV file: %s", input_path)

    # Detect how many rows to skip to get to actual data
    csv_info = sniff_csv_metadata(input_path)

    # Read CSV with pandas; let it handle encoding automatically
    log.warning("Reading CSV file, skipping %d rows", csv_info.skip_rows)
    df = pd.read_csv(
        input_path,
        skiprows=csv_info.skip_rows,
        engine="python",
        on_bad_lines="skip",
        encoding_errors="replace",
    )
    log.info("Successfully read CSV with shape: %s", df.shape)

    log.info("Columns found: %d", len(df.columns))

    # Check which target columns exist
    existing_columns = []
    missing_columns = []

    for col in target_columns:
        if col in df.columns:
            existing_columns.append(col)
        else:
            missing_columns.append(col)

    log.info("Found %d of %d target columns", len(existing_columns), len(target_columns))

    if missing_columns:
        log.error("Missing columns:\n%s", "\n".join(f"  - {col!r}" for col in missing_columns))

        # Try to find similar column names
        for missing_col in missing_columns:
            similar_cols = [
                col
                for col in df.columns
                if missing_col.lower() in col.lower() or col.lower() in missing_col.lower()
            ]
            if similar_cols:
                log.info("For %r, found similar: %r", missing_col, similar_cols)

    if not existing_columns:
        raise ValueError("No target columns found! Cannot proceed.")

    # Extract the existing columns
    filtered_df = df[existing_columns].copy()

    # Truncate to max_rows if specified (0 means no limit)
    if max_rows > 0 and len(filtered_df) > max_rows:
        log.info("Truncating data from %d to %d rows", len(filtered_df), max_rows)
        filtered_df = filtered_df.head(max_rows)

    # Apply column transformations
    if column_transformations:
        for column_name, transform_func in column_transformations.items():
            if column_name in filtered_df.columns:
                log.info("Applying transformation to column: %s", column_name)
                column_series = filtered_df[column_name].fillna("").astype(str)  # pyright: ignore
                filtered_df[column_name] = column_series.apply(transform_func)
                log.info(
                    "Column '%s' values after transformation: %s",
                    column_name,
                    filtered_df[column_name].value_counts().to_dict(),  # pyright: ignore
                )

    log.info("Filtered data shape: %s", filtered_df.shape)
    log.info("Non-null values per column:")
    for col in filtered_df.columns:
        non_null_count = filtered_df[col].count()  # pyright: ignore
        total_count = len(filtered_df)
        log.info(
            "  %s: %d/%d (%.1f%%)",
            col,
            non_null_count,
            total_count,
            non_null_count / total_count * 100,
        )

    # Write to output file
    filtered_df.to_csv(output_path, index=False)


def simplify_sentiment(value: str) -> str:
    if "more excited" in value.lower():
        return "excited"
    elif "more concerned" in value.lower():
        return "concerned"
    elif "equally" in value.lower():
        return "neutral"
    else:
        return ""


# Participants data.
# NB: keep the typo "particpants"! (not "participants").
CSV_URL = "https://huggingface.co/datasets/collective-intelligence-project/Global-AI-Dialogues/raw/main/Global%20AI%20Dialogues%20Data%20-%20September%202024/particpants.csv"
TARGET_COLUMNS = [
    "Overall, would you say the increased use of artificial intelligence (AI) in daily life makes you feel…",
    "What do you think your life might be like in 30 years? Alt: Imagine life 30 years from now. What's the biggest difference you notice in daily life compared to today? (English)",
    "What is your gender?",
    "How old are you?",
    "What country or region do you most identify with?",
    "What religious group or faith do you most identify with?",
    # "Participant Id",
]
COLUMN_TRANSFORMATIONS: dict[str, Callable[[str], str]] = {
    "Overall, would you say the increased use of artificial intelligence (AI) in daily life makes you feel…": simplify_sentiment,
}


@kash_action(
    params=(Param("max_rows", "Maximum number of rows to include in the visualization", type=int),)
)
def gd_csv_simplify_participants(item: Item, max_rows: int = 0) -> Item:
    """
    Clean up/simplify the Global Dialogues participant CSV file.
    """
    ws = current_ws()
    assert item.store_path

    simplified_data = item.derived_copy(title="participants_simple")
    target_path = ws.target_path_for(simplified_data)

    log.warning("Simplifying data to: %s", target_path)

    simplify_csv(
        ws.base_dir / item.store_path, target_path, TARGET_COLUMNS, max_rows, COLUMN_TRANSFORMATIONS
    )
    simplified_data.external_path = str(target_path)

    log.warning("Simplified data: %s", simplified_data)

    return simplified_data


## Tests


def test_simplify_sentiment() -> None:
    """Test the simplify_sentiment function."""
    assert simplify_sentiment("more excited") == "excited"
    assert simplify_sentiment("MORE EXCITED") == "excited"
    assert simplify_sentiment("more concerned") == "concerned"
    assert simplify_sentiment("MORE CONCERNED") == "concerned"
    assert simplify_sentiment("equally") == "neutral"
    assert simplify_sentiment("EQUALLY") == "neutral"
    assert simplify_sentiment("something else") == ""
    assert simplify_sentiment("") == ""


if __name__ == "__main__":
    test_simplify_sentiment()
    print("✅ All tests passed!")
