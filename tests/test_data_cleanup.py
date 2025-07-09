from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from global_dialogues_viz.data_cleanup import simplify_csv


def test_simplify_csv_basic():
    """Test basic functionality of simplify_csv function"""
    # Create a temporary CSV file with test data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Name,Age,City,Country\n")
        f.write("John,25,NYC,USA\n")
        f.write("Jane,30,Paris,France\n")
        f.write("Bob,35,London,UK\n")
        input_path = Path(f.name)

    # Create output path
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_path = Path(f.name)

    try:
        # Define target columns
        target_columns = ["Name", "Age", "Country"]

        # Test the function
        simplify_csv(input_path, output_path, target_columns)

        # Read the output and verify
        df = pd.read_csv(output_path)
        assert list(df.columns) == target_columns
        assert len(df) == 3
        assert df["Name"].tolist() == ["John", "Jane", "Bob"]
        assert df["Age"].tolist() == [25, 30, 35]
        assert df["Country"].tolist() == ["USA", "France", "UK"]

    finally:
        # Clean up temporary files
        input_path.unlink()
        output_path.unlink()


def test_simplify_csv_missing_columns():
    """Test simplify_csv when some target columns are missing"""
    # Create a temporary CSV file with test data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Name,Age,City\n")
        f.write("John,25,NYC\n")
        f.write("Jane,30,Paris\n")
        input_path = Path(f.name)

    # Create output path
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_path = Path(f.name)

    try:
        # Define target columns (including missing "Country")
        target_columns = ["Name", "Age", "Country"]

        # Test the function - should work with available columns
        simplify_csv(input_path, output_path, target_columns)

        # Read the output and verify only available columns are present
        df = pd.read_csv(output_path)
        assert list(df.columns) == ["Name", "Age"]
        assert len(df) == 2

    finally:
        # Clean up temporary files
        input_path.unlink()
        output_path.unlink()


def test_simplify_csv_no_target_columns():
    """Test simplify_csv when no target columns are found"""
    # Create a temporary CSV file with test data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Name,Age,City\n")
        f.write("John,25,NYC\n")
        input_path = Path(f.name)

    # Create output path
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_path = Path(f.name)

    try:
        # Define target columns that don't exist
        target_columns = ["Country", "Phone"]

        # Test the function - should raise ValueError
        with pytest.raises(ValueError, match="No target columns found"):
            simplify_csv(input_path, output_path, target_columns)

    finally:
        # Clean up temporary files
        input_path.unlink()
        if output_path.exists():
            output_path.unlink()


def test_simplify_csv_with_max_rows(tmp_path: Path) -> None:
    """Test that simplify_csv correctly truncates data to max_rows."""
    # Create a test CSV with more rows than max_rows
    csv_content = """Header1,Header2,Header3
Row1Col1,Row1Col2,Row1Col3
Row2Col1,Row2Col2,Row2Col3
Row3Col1,Row3Col2,Row3Col3
Row4Col1,Row4Col2,Row4Col3
Row5Col1,Row5Col2,Row5Col3
"""

    input_file = tmp_path / "test_input.csv"
    input_file.write_text(csv_content)

    output_file = tmp_path / "test_output.csv"

    # Test with max_rows=3
    simplify_csv(input_file, output_file, ["Header1", "Header2"], max_rows=3)

    # Read the output and verify it has exactly 3 rows (plus header)
    output_content = output_file.read_text()
    lines = output_content.strip().split("\n")

    # Should have header + 3 data rows = 4 lines total
    assert len(lines) == 4
    assert lines[0] == "Header1,Header2"
    assert lines[1] == "Row1Col1,Row1Col2"
    assert lines[2] == "Row2Col1,Row2Col2"
    assert lines[3] == "Row3Col1,Row3Col2"


def test_simplify_csv_max_rows_zero_means_no_limit(tmp_path: Path) -> None:
    """Test that max_rows=0 means no limit."""
    # Create a test CSV with 5 rows
    csv_content = """Header1,Header2,Header3
Row1Col1,Row1Col2,Row1Col3
Row2Col1,Row2Col2,Row2Col3
Row3Col1,Row3Col2,Row3Col3
Row4Col1,Row4Col2,Row4Col3
Row5Col1,Row5Col2,Row5Col3
"""

    input_file = tmp_path / "test_input.csv"
    input_file.write_text(csv_content)

    output_file = tmp_path / "test_output.csv"

    # Test with max_rows=0 (no limit)
    simplify_csv(input_file, output_file, ["Header1", "Header2"], max_rows=0)

    # Read the output and verify it has all 5 rows (plus header)
    output_content = output_file.read_text()
    lines = output_content.strip().split("\n")

    # Should have header + 5 data rows = 6 lines total
    assert len(lines) == 6
    assert lines[0] == "Header1,Header2"
    assert lines[1] == "Row1Col1,Row1Col2"
    assert lines[2] == "Row2Col1,Row2Col2"
    assert lines[3] == "Row3Col1,Row3Col2"
    assert lines[4] == "Row4Col1,Row4Col2"
    assert lines[5] == "Row5Col1,Row5Col2"
