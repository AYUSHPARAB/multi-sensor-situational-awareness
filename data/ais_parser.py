"""
AIS Data Parser for Danish Maritime Authority Open Data
=======================================================
Parses raw AIS CSV files from https://web.ais.dk/aisdata/
Filters to a geographic bounding box, cleans nulls, and outputs
a structured DataFrame ready for downstream fusion.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_BBOX = {
    "lat_min": 54.5,
    "lat_max": 57.5,
    "lon_min": 9.0,
    "lon_max": 13.0,
}


OUTPUT_COLUMNS = [
    "timestamp",
    "mmsi",
    "latitude",
    "longitude",
    "speed_over_ground",
    "course_over_ground",
    "heading",
    "vessel_type",
    "vessel_name",
    "navigational_status",
    "rate_of_turn",
    "ship_length",
    "ship_width",
    "draught",
]


class AISParser:
    """Parses and cleans Danish Maritime Authority AIS data."""

    def __init__(
            self,
            bbox: tuple[float, float, float, float] | None = None,
            min_speed: float = 0.0,
            max_speed: float = 50.0   
    ):
        if bbox:
            self.lat_min, self.lat_max, self.lon_min, self.lon_max = bbox
        else:
            self.lat_min = DEFAULT_BBOX["lat_min"]
            self.lat_max = DEFAULT_BBOX["lat_max"]
            self.lon_min = DEFAULT_BBOX["lon_min"]
            self.lon_max = DEFAULT_BBOX["lon_max"]

        self.min_speed = min_speed
        self.max_speed = max_speed


    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map DMA CSV column names to clean snake_case names."""
        column_map = {
            "# Timestamp": "timestamp",
            "Timestamp": "timestamp",
            "MMSI": "mmsi",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Navigational status": "navigational_status",
            "ROT": "rate_of_turn",
            "SOG": "speed_over_ground",
            "COG": "course_over_ground",
            "Heading": "heading",
            "Name": "vessel_name",
            "Ship type": "vessel_type",
            "Width": "ship_width",
            "Length": "ship_length",
            "Draught": "draught",
        }
        return df.rename(columns=column_map)


    def parse(
            self,
            filepath: str,
            nrows: int | None = None,
            sample_vessels: int | None = None
    ) -> pd.DataFrame:
        filepath = Path(filepath)
        logger.info(f"Parsing AIS data from {filepath}...")

        #reading csv
        df = pd.read_csv(
            filepath,
            nrows=nrows,
            low_memory=False,
            na_values=["Unknown", "undefined", ""],
        )
        logger.info(f" Raw rows loaded: {len(df)}")

        #renaiming columns
        df = self._rename_columns(df)

        #converting types
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            format="mixed",
            dayfirst=True,
        )

        numeric_cols = [
            "latitude",
            "longitude",
            "speed_over_ground",
            "course_over_ground",
            "heading",
            "rate_of_turn",
            "ship_length",
            "ship_width",
            "draught"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["mmsi"] = df["mmsi"].astype(str).str.strip()

        #dropping rows with missing values.
        df = df.dropna(subset=["latitude", "longitude", "mmsi", "timestamp"])
        logger.info(f" After dropping nulls: {len(df):,}")

        #dropping invalid coordinates
        df = df [
            (df["latitude"].between(-90,90))
            & (df["longitude"].between(-180,180))
        ]
        logger.info(f" After coordinate validation: {len(df):,}")

        #Applying bbox filter
        df = df[
            (df["latitude"].between(self.lat_min, self.lat_max))
            & (df["longitude"].between(self.lon_min, self.lon_max))
        ]
        logger.info(
            f" After Bbox filter "
            f"[{self.lat_min}-{self.lat_max}N, "
            f"{self.lon_min}-{self.lon_max}E]: {len(df):,}"
        )

        #Applying speed filter
        if "speed_over_ground" in df.columns:
            df = df[
                (df["speed_over_ground"] >= self.min_speed)
                & (df["speed_over_ground"] <= self.max_speed)
            ]
            logger.info(
                f"  After speed filter "
                f"[{self.min_speed}-{self.max_speed} kn]: {len(df):,}"
            )

        #Removing duplicate readings.
        df = df.drop_duplicates(subset=["mmsi", "timestamp"])
        logger.info(f"  After deduplication: {len(df):,}")

        #sorting per vessel timestamp
        df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

        #Sample vessels, if requested.
        if sample_vessels and df["mmsi"].nunique() > sample_vessels:
            sampled = np.random.choice(
                df["mmsi"].unique(),
                size=sample_vessels,
                replace=False,
            )
            df = df[df["mmsi"].isin(sampled)]
            logger.info(
                f"  Sampled {sample_vessels} vessels: {len(df):,} rows"
            )

        #store only output columns.
        available = [c for c in OUTPUT_COLUMNS if c in df.columns]
        df = df[available]

        logger.info(
            f"  Final: {len(df):,} rows | "
            f"{df['mmsi'].nunique()} vessels | "
            f"{df['timestamp'].min()} to {df['timestamp'].max()}"
        )


        return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Generate summary statistics for parsed AIS data."""
    return {
        "total_rows": len(df),
        "unique_vessels": df["mmsi"].nunique(),
        "time_range": (
            f"{df['timestamp'].min()} → {df['timestamp'].max()}"
        ),
        "duration_hours": round(
            (df["timestamp"].max() - df["timestamp"].min())
            .total_seconds() / 3600,
            1,
        ),
        "avg_speed_kn": (
            round(df["speed_over_ground"].mean(), 2)
            if "speed_over_ground" in df.columns
            else None
        ),
        "bbox": {
            "lat": (df["latitude"].min(), df["latitude"].max()),
            "lon": (df["longitude"].min(), df["longitude"].max()),
        },
        "vessel_types": (
            df["vessel_type"].value_counts().head(5).to_dict()
            if "vessel_type" in df.columns
            else {}
        ),
    }


if __name__ == "__main__":
    parser_cli = argparse.ArgumentParser(
        description="Parse Danish Maritime Authority AIS data"
    )
    parser_cli.add_argument(
        "--input", "-i",
        required=True,
        help="Path to raw AIS CSV file from web.ais.dk",
    )
    parser_cli.add_argument(
        "--output", "-o",
        default="data/processed/ais_clean.csv",
        help="Output path for cleaned CSV",
    )
    parser_cli.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limit rows read (for testing)",
    )
    parser_cli.add_argument(
        "--sample-vessels",
        type=int,
        default=None,
        help="Randomly sample N vessels",
    )
    args = parser_cli.parse_args()

    # Runs the parser.
    ais = AISParser()
    df = ais.parse(
        args.input,
        nrows=args.nrows,
        sample_vessels=args.sample_vessels,
    )

    # Saving the output.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    # Print the summary.
    stats = get_summary_stats(df)
    print("\n" + "=" * 50)
    print("AIS DATA SUMMARY")
    print("=" * 50)
    for key, value in stats.items():
        print(f"  {key:<20}: {value}")
    print("=" * 50)
