"""Normalize a prediction CSV to the Kaggle submission schema.

Kaggle requires exactly: id,total_sales
"""
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = ["id", "total_sales"]


def normalize_submission(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path

    submission = pd.read_csv(input_path)

    if "predicted_sales" in submission.columns and "total_sales" not in submission.columns:
        submission = submission.rename(columns={"predicted_sales": "total_sales"})

    if list(submission.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            f"Invalid submission columns: {list(submission.columns)}. "
            f"Expected exactly {REQUIRED_COLUMNS}."
        )

    if submission[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Submission contains missing IDs or predictions.")

    submission.to_csv(output_path, index=False)
    print(f"Saved valid Kaggle submission: {output_path}")
    return output_path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    normalize_submission(
        repo_root / "submissions" / "submission_final.csv",
        repo_root / "submissions" / "submission_kaggle.csv",
    )
