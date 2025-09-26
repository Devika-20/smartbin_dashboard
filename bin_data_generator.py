import json, random, math
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BINS_MASTER_PATH = DATA_DIR / "bins_master.csv"
HISTORY_PATH = DATA_DIR / "history.csv"
WEATHER_PROFILES = DATA_DIR / "weather_profiles.json"

RNG = np.random.default_rng(42)

def ensure_bins_master(n_bins: int = 60):
    if BINS_MASTER_PATH.exists():
        return pd.read_csv(BINS_MASTER_PATH)
    # Make bins around a city center with lat/lon jitter
    base_lat, base_lon = 13.0827, 80.2707  # Chennai-ish as placeholder; change to your city
    rows = []
    for i in range(1, n_bins + 1):
        lat = base_lat + RNG.normal(0, 0.05)
        lon = base_lon + RNG.normal(0, 0.05)
        area_type = RNG.choice(["residential", "commercial", "market", "institutional"], p=[0.45, 0.25, 0.2, 0.1])
        nearby_pop = int(max(200, RNG.normal(1500 if area_type != "market" else 2500, 300)))
        capacity_l = RNG.choice([240, 360, 660, 1100], p=[0.3, 0.35, 0.25, 0.1])
        rows.append({"bin_id": f"B{i:03d}", "lat": lat, "lon": lon, "area_type": area_type, "nearby_population": nearby_pop, "capacity_l": capacity_l})
    df = pd.DataFrame(rows)
    df.to_csv(BINS_MASTER_PATH, index=False)
    return df

def ensure_weather_profiles():
    if WEATHER_PROFILES.exists():
        return json.loads(WEATHER_PROFILES.read_text())
    profiles = {
        "sunny": {"temp_mean": 33, "temp_std": 3, "effect": 1.0},
        "cloudy": {"temp_mean": 30, "temp_std": 2, "effect": 0.95},
        "rainy": {"temp_mean": 27, "temp_std": 2, "effect": 0.8},
        "festival": {"temp_mean": 31, "temp_std": 2, "effect": 1.25}
    }
    WEATHER_PROFILES.write_text(json.dumps(profiles, indent=2))
    return profiles

def simulate_history(days: int = 30, bins_per_city: int = 60, seed: int = 123):
    RNG = np.random.default_rng(seed)
    bins_df = ensure_bins_master(bins_per_city)
    weather_profiles = ensure_weather_profiles()

    records = []
    start = datetime.utcnow() - timedelta(days=days)
    for b in bins_df.itertuples(index=False):
        fill = RNG.uniform(0, 40)  # start from 0-40%
        for d in range(days * 24):  # hourly
            ts = start + timedelta(hours=d)
            dow = ts.weekday()
            hour = ts.hour
            # Weather & events
            weather = RNG.choice(["sunny", "cloudy", "rainy"], p=[0.55, 0.3, 0.15])
            # Add occasional festival spikes
            festival = 1 if RNG.random() < 0.03 else 0
            if festival:
                weather = "festival"
            prof = weather_profiles[weather]
            temp = RNG.normal(prof["temp_mean"], prof["temp_std"])

            # Base fill rate influenced by area, hour, population, and weather profile
            hour_factor = 1.2 if 8 <= hour <= 11 else (1.3 if 18 <= hour <= 21 else 0.7)
            area_mult = {"residential": 1.0, "commercial": 1.15, "market": 1.4, "institutional": 0.9}[b.area_type]
            pop_effect = min(2.0, 0.5 + b.nearby_population / 2000.0)
            weather_effect = prof["effect"]
            base_rate = 1.0 * hour_factor * area_mult * pop_effect * weather_effect

            # Add noise
            rate = RNG.normal(base_rate, 0.35)
            rate = max(0, rate)

            # Occasional collection event (empties bin)
            collection = 1 if RNG.random() < 0.06 else 0
            if collection or fill >= 95:
                fill = RNG.uniform(5, 15)
                collected = 1
            else:
                fill = min(100.0, fill + rate)
                collected = 0

            overflow = 1 if fill >= 90 else 0

            records.append({
                "ts": ts.isoformat(),
                "bin_id": b.bin_id,
                "lat": b.lat,
                "lon": b.lon,
                "area_type": b.area_type,
                "nearby_population": b.nearby_population,
                "capacity_l": b.capacity_l,
                "temp_c": float(temp),
                "weather": weather,
                "hour": hour,
                "dow": dow,
                "festival": festival,
                "fill_level": float(fill),
                "collected": collected,
                "overflow": overflow,
            })

    df = pd.DataFrame(records)
    df.to_csv(HISTORY_PATH, index=False)
    return df

if __name__ == "__main__":
    ensure_bins_master()
    ensure_weather_profiles()
    simulate_history()
    print("Synthetic data generated at", DATA_DIR)
