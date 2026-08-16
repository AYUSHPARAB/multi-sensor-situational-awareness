"""
Sample AIS Data Generator
=========================
Generates realistic AIS data for testing when real
Danish Maritime Authority data is unavailable.
Replace with real data from http://aisdata.ais.dk/ for production.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def generate_sample_ais(
    n_vessels: int = 30,
    n_hours: int = 6,
    output_path: str = "data/raw/sample_ais.csv",
    seed: int = 42,
) -> str:
    """
    Generate realistic sample AIS data for the Danish Straits.

    Parameters
    ----------
    n_vessels : int
        Number of vessels to simulate.
    n_hours : int
        Duration of simulation in hours.
    output_path : str
        Where to save the generated CSV.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    str
        Path to the generated CSV file.
    """
    logger.info(
        f"Generating sample AIS data "
        f"({n_vessels} vessels, {n_hours} hours)..."
    )

    rng = np.random.default_rng(seed)
    records = []
    base_time = pd.Timestamp("2025-01-15 06:00:00")

    vessel_names = [
        "NORDIC STAR", "BALTIC TRADER", "COPENHAGEN EXPRESS",
        "MAERSK AURORA", "VIKING SPIRIT", "KATTEGAT FERRY",
        "ORESUND CARRIER", "SKAGEN FISHER", "JUTLAND PRIDE",
        "BELT RUNNER", "STOREBALT", "KRONBORG", "ELSINORE",
        "HAMLET", "FREYA", "ODIN", "THOR", "AEGIR", "NJORD",
        "SIGRID", "BJORN", "ASTRID", "RAGNHILD", "SVENBORG",
        "AALBORG STAR", "ESBJERG TRADER", "HELSINGOR",
        "KORSOR", "NYBORG", "ODENSE WAVE",
    ]

    vessel_types = ["Cargo", "Tanker", "Passenger", "Fishing", "Tug"]

    for v in range(n_vessels):
        mmsi = f"21900{v:04d}"
        name = vessel_names[v % len(vessel_names)]
        vessel_type = rng.choice(vessel_types)

        # Random starting position in Danish Straits bounding box
        lat = rng.uniform(54.8, 56.5)
        lon = rng.uniform(10.0, 12.5)

        # Random movement characteristics
        speed = rng.uniform(5, 20)
        heading = rng.uniform(0, 360)

        # One position every 30 seconds
        interval_seconds = 30
        total_steps = int(n_hours * 3600 / interval_seconds)

        for step in range(total_steps):
            timestamp = base_time + pd.Timedelta(
                seconds=step * interval_seconds
            )

            # Small random variations each step
            speed = max(0, speed + rng.normal(0, 0.3))
            heading = (heading + rng.normal(0, 1)) % 360

            # Move vessel
            dt_hours = interval_seconds / 3600
            lat += (
                (speed * dt_hours / 60)
                * np.cos(np.radians(heading))
            )
            lon += (
                (speed * dt_hours / 60)
                * np.sin(np.radians(heading))
                / np.cos(np.radians(lat))
            )

            records.append({
                "# Timestamp": timestamp.strftime("%d/%m/%Y %H:%M:%S"),
                "Type of mobile": "Class A",
                "MMSI": mmsi,
                "Latitude": round(lat, 6),
                "Longitude": round(lon, 6),
                "Navigational status": "Under way using engine",
                "ROT": round(rng.normal(0, 2), 1),
                "SOG": round(speed, 1),
                "COG": round(heading, 1),
                "Heading": round(heading, 0),
                "IMO": f"IMO{9000000 + v}",
                "Callsign": f"OX{v:04d}",
                "Name": name,
                "Ship type": vessel_type,
                "Cargo type": "",
                "Width": int(rng.choice([15, 20, 25, 30, 32, 40])),
                "Length": int(rng.choice([80, 120, 150, 180, 200, 250])),
                "Type of position fixing device": "GPS",
                "Draught": round(rng.uniform(3, 12), 1),
                "Destination": rng.choice([
                    "COPENHAGEN", "HAMBURG", "GOTHENBURG", "AARHUS"
                ]),
                "ETA": "",
                "Data source type": "AIS",
            })

    df = pd.DataFrame(records)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"  Generated {len(df):,} rows")
    logger.info(f"  {df['MMSI'].nunique()} unique vessels")
    logger.info(f"  Saved to {output_path}")

    return str(output_path)


if __name__ == "__main__":
    path = generate_sample_ais(
        n_vessels=30,
        n_hours=6,
        output_path="data/raw/sample_ais.csv",
    )
    print(f"\nSample data ready at: {path}")
    print("Use this to test the pipeline.")
    print("Replace with real DMA data from http://aisdata.ais.dk/")
    