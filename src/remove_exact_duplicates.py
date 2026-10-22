from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "processed_data" / "clean_dos_scenario"
OUTPUT_ROOT = ROOT / "processed_data" / "preprocessed_deduplicated"
SPLITS = ("train", "validation", "test")
SCENARIOS = ("highway_2", "highway_7")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "EXACT DUPLICATE-ROW REMOVAL",
        "Original files are preserved.",
        f"Output directory: {OUTPUT_ROOT.relative_to(ROOT)}",
        "",
    ]

    for scenario in SCENARIOS:
        for split in SPLITS:
            source = INPUT_ROOT / scenario / f"{split}.parquet"
            if not source.exists():
                raise FileNotFoundError(f"Required input file not found: {source}")

            # Process one file at a time to limit memory use.
            df = pd.read_parquet(source)
            rows_before = len(df)
            df_clean = df.drop_duplicates(keep="first")
            rows_after = len(df_clean)
            removed = rows_before - rows_after

            destination_dir = OUTPUT_ROOT / scenario
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"{split}.parquet"
            df_clean.to_parquet(destination, index=False, engine="pyarrow")

            line = (
                f"{scenario}/{split}: {rows_before:,} rows -> "
                f"{rows_after:,} rows; removed {removed:,} exact duplicate rows."
            )
            print(line)
            report_lines.append(line)

            del df, df_clean

    report_path = OUTPUT_ROOT / "duplicate_removal_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nReport saved to: {report_path.relative_to(ROOT)}")
    print("Done. Original clean datasets were not changed.")


if __name__ == "__main__":
    main()
