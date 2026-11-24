"""Convert the raw Excel export into the anonymised CSV released in ``data/``.

Only the sheet ``realdata`` is read. Every other sheet (participant metadata
such as names, age or gender) is ignored and never written to the output.

Usage:
    python scripts/prepare_data.py real_data_consolidated_raw.xlsx data/cabin_comfort.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RENAME = {
    "Time": "timestamp",
    "User_id": "driver_code",
    "Ambient Temperature": "ambient_temp_c",
    "Cabin Temperature": "cabin_temp_c",
    "Relative Humidity": "cabin_rh_pct",
    "Solar Load": "light_intensity",
    "Metabolic rate": "met",
    "Clothing insulation": "clo",
    "Zone1 skin Temperature": "z1_skin_temp_c",
    "Zone1 RH": "z1_rh_pct",
    "Zone2 skin Temperature": "z2_skin_temp_c",
    "Zone2 RH": "z2_rh_pct",
    "Zone3 skin Temperature": "z3_skin_temp_c",
    "Zone3 RH": "z3_rh_pct",
    "Zone1 PMV (driver)": "z1_vote",
    "Zone2 PMV (co-pas)": "z2_vote",
    "Zone3_L PMV": "z3l_vote",
    "Zone3_R PMV": "z3r_vote",
}
VOTES = ["z1_vote", "z2_vote", "z3l_vote", "z3r_vote"]
OUTPUT_COLUMNS = [
    "timestamp",
    "day",
    "drive_id",
    "driver_id",
    "elapsed_s",
    "ambient_temp_c",
    "cabin_temp_c",
    "cabin_rh_pct",
    "light_intensity",
    "met",
    "clo",
    "z1_skin_temp_c",
    "z1_rh_pct",
    "z2_skin_temp_c",
    "z2_rh_pct",
    "z3_skin_temp_c",
    "z3_rh_pct",
    *VOTES,
]


def convert(xlsx: Path, out: Path, day_gap_s: float = 3600.0) -> pd.DataFrame:
    raw = pd.read_excel(xlsx, sheet_name="realdata")
    raw = raw.loc[:, [c for c in raw.columns if not str(c).startswith("Unnamed")]]
    unknown = sorted(set(raw.columns) - set(RENAME))
    if unknown:
        raise ValueError(f"unexpected columns in sheet 'realdata': {unknown}")
    df = raw.rename(columns=RENAME)

    # '-' marks an empty seat (rear right on day 2).
    for col in VOTES:
        df[col] = pd.to_numeric(df[col].replace("-", np.nan), errors="coerce")

    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    gap = df["timestamp"].diff().dt.total_seconds()
    df["day"] = (gap > day_gap_s).cumsum() + 1
    new_drive = (df["driver_code"] != df["driver_code"].shift()) | (gap > day_gap_s)
    df["drive_id"] = new_drive.cumsum()
    # Pseudonymous driver index (1, 2, ...) in order of first appearance.
    codes = list(dict.fromkeys(df["driver_code"]))
    df["driver_id"] = df["driver_code"].map({c: i + 1 for i, c in enumerate(codes)})
    first = df.groupby("drive_id")["timestamp"].transform("min")
    df["elapsed_s"] = (df["timestamp"] - first).dt.total_seconds().round(3)

    df = df[OUTPUT_COLUMNS]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6g", date_format="%Y-%m-%dT%H:%M:%S.%f")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("xlsx", type=Path)
    parser.add_argument("out", type=Path, nargs="?", default=Path("data/cabin_comfort.csv"))
    args = parser.parse_args()
    df = convert(args.xlsx, args.out)
    summary = df.groupby("drive_id").agg(
        day=("day", "first"),
        driver=("driver_id", "first"),
        samples=("timestamp", "size"),
        start=("timestamp", "min"),
        end=("timestamp", "max"),
    )
    print(f"wrote {len(df)} rows to {args.out}\n{summary}")


if __name__ == "__main__":
    main()
