import pandas as pd
import numpy as np
from pyproj import Transformer
from scipy.spatial import KDTree
from pathlib import Path

# Paths are resolved relative to this script file, not the working directory
BASE = Path(__file__).parent.parent.parent / "data" / "processed"

dataset1 = pd.read_csv(BASE / "datasets_cleaned" / "dataset1_cleaned.csv")
dataset2 = pd.read_csv(BASE / "datasets_cleaned" / "dataset2_cleaned.csv")
dataset6 = pd.read_csv(BASE / "dataset6_cleaned.csv")

# --- Normalize borough names ---
borough_map = {
    "Manhattan": "Manhattan", "New York": "Manhattan",
    "Bronx": "Bronx", "The Bronx": "Bronx", "Bronx County": "Bronx",
    "Brooklyn": "Brooklyn", "Queens": "Queens",
    "Staten Island": "Staten Island", "Staten island": "Staten Island"
}
for df in [dataset1, dataset2, dataset6]:
    borough_col = "Boro" if "Boro" in df.columns else "borough"
    df[borough_col] = df[borough_col].astype(str).str.strip().str.title()
    df[borough_col] = df[borough_col].map(borough_map).fillna(df[borough_col])

dataset6 = dataset6.rename(columns={"Boro": "borough"})

# --- Step 1: Convert dataset6 WktGeom (NY State Plane EPSG:2263) to lat/lon ---
transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)

def parse_wkt_to_latlon(wkt):
    x, y = wkt.replace("POINT (", "").replace(")", "").split()
    lon, lat = transformer.transform(float(x), float(y))
    return lat, lon

dataset6[["lat", "lon"]] = dataset6["WktGeom"].apply(
    lambda w: pd.Series(parse_wkt_to_latlon(w), index=["lat", "lon"])
)

# --- Step 2: Compute centroid lat/lon for each unique link in dataset2 ---
def get_link_centroid(link_points_str):
    lats, lons = [], []
    for pair in str(link_points_str).strip().split():
        parts = pair.split(",")
        if len(parts) == 2:
            try:
                lats.append(float(parts[0]))
                lons.append(float(parts[1]))
            except ValueError:
                continue
    if lats:
        return np.mean(lats), np.mean(lons)
    return np.nan, np.nan

unique_links = dataset2[["link_name", "link_points", "borough"]].drop_duplicates("link_name").copy()
unique_links[["cent_lat", "cent_lon"]] = unique_links["link_points"].apply(
    lambda lp: pd.Series(get_link_centroid(lp), index=["cent_lat", "cent_lon"])
)
unique_links = unique_links.dropna(subset=["cent_lat", "cent_lon"]).reset_index(drop=True)

# --- Step 3: Match each dataset6 segment to its nearest link using a KDTree ---
tree = KDTree(unique_links[["cent_lat", "cent_lon"]].values)
distances, indices = tree.query(dataset6[["lat", "lon"]].values)

dataset6["link_name"] = unique_links["link_name"].iloc[indices].values
dataset6["match_distance_deg"] = distances

# --- Step 4: Merge dataset1 and dataset2 at row level on link_name + borough ---
ds1_ds2 = pd.merge(dataset1, dataset2, on=["link_name", "borough"], how="inner")

# --- Step 5: Extract time components from dataset2's per-row timestamp ---
ds1_ds2["data_as_of"] = pd.to_datetime(ds1_ds2["data_as_of"])
ds1_ds2["Yr"] = ds1_ds2["data_as_of"].dt.year
ds1_ds2["M"]  = ds1_ds2["data_as_of"].dt.month
ds1_ds2["D"]  = ds1_ds2["data_as_of"].dt.day
ds1_ds2["HH"] = ds1_ds2["data_as_of"].dt.hour

# --- Step 6: Aggregate ds1+ds2 to link + hour level ---
# Each unique link-hour bucket gets a single average value for speed, travel time, etc.
# Aggregating by link_name (not just borough) gives per-segment averages
# rather than borough-wide averages, preserving much more geographic detail.
ds1_ds2_agg = ds1_ds2.groupby(
    ["link_name", "borough", "Yr", "M", "D", "HH"], as_index=False
).mean(numeric_only=True)

# --- Step 7: Join dataset6 (volume) with aggregated speed/travel time on link + time ---
merged = pd.merge(
    dataset6,
    ds1_ds2_agg,
    on=["link_name", "Yr", "M", "D", "HH"],
    how="left"
)

# --- Step 8: Resolve duplicate columns created by the merge ---
# borough exists in both dataset6 and ds1_ds2_agg, so pandas renames them
# to borough_x/borough_y. Consolidate back to a single column,
# preferring dataset6's values (borough_x) as the primary source.
if "borough_x" in merged.columns:
    merged["borough"] = merged["borough_x"].fillna(merged["borough_y"])
    merged = merged.drop(columns=["borough_x", "borough_y"])

# --- Step 9: Sort for readability ---
merged = merged.sort_values(
    by=["borough", "street", "fromSt", "toSt", "Yr", "M", "D", "HH", "MM"]
)
merged = merged.reset_index(drop=True)

print("Shape:", merged.shape)
print(merged.head())
print("\nNull counts:\n", merged.isnull().sum())
print("\nSample match distances (degrees):\n", dataset6["match_distance_deg"].describe())

merged.to_csv("merged_dataset_aggregated.csv", index=False)
