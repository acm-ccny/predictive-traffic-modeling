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
# WktGeom coordinates are in feet using the NY State Plane projection.
# always_xy=True ensures output is always (lon, lat) regardless of CRS axis order.
transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)

def parse_wkt_to_latlon(wkt):
    x, y = wkt.replace("POINT (", "").replace(")", "").split()
    lon, lat = transformer.transform(float(x), float(y))
    return lat, lon

dataset6[["lat", "lon"]] = dataset6["WktGeom"].apply(
    lambda w: pd.Series(parse_wkt_to_latlon(w), index=["lat", "lon"])
)

# --- Step 2: Compute centroid lat/lon for each unique link in dataset2 ---
# link_points is a space-separated string of "lat,lon" pairs defining the link geometry.
# The centroid is the average position, used as the representative point for matching.
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
# KDTree performs fast nearest-neighbor lookup on the centroid coordinates.
# Each dataset6 point gets assigned the link_name of the closest link centroid.
tree = KDTree(unique_links[["cent_lat", "cent_lon"]].values)
distances, indices = tree.query(dataset6[["lat", "lon"]].values)

dataset6["link_name"] = unique_links["link_name"].iloc[indices].values
dataset6["match_distance_deg"] = distances  # useful for spot-checking match quality

# --- Step 4: Merge dataset1 and dataset2 at row level on link_name + borough ---
# No aggregation — every individual link-level reading is preserved.
ds1_ds2 = pd.merge(dataset1, dataset2, on=["link_name", "borough"], how="outer")

# --- Step 5: Extract time components from dataset2's per-row timestamp ---
ds1_ds2["data_as_of"] = pd.to_datetime(ds1_ds2["data_as_of"])
ds1_ds2["Yr"] = ds1_ds2["data_as_of"].dt.year
ds1_ds2["M"]  = ds1_ds2["data_as_of"].dt.month
ds1_ds2["D"]  = ds1_ds2["data_as_of"].dt.day
ds1_ds2["HH"] = ds1_ds2["data_as_of"].dt.hour
ds1_ds2["MM"] = ds1_ds2["data_as_of"].dt.minute

# --- Step 6: Join dataset6 with ds1+ds2 on link_name + time ---
# Now that dataset6 has a matched link_name, we join on the actual segment ID
# plus time, giving each volume row the speed/travel time for that exact link and hour.
merged = pd.merge(
    dataset6,
    ds1_ds2,
    on=["link_name", "Yr", "M", "D", "HH"],
    how="outer"
)

# --- Step 7: Resolve duplicate columns created by the outer merge ---
# borough and MM exist in both dataset6 and ds1_ds2, so pandas renames them
# to borough_x/borough_y and MM_x/MM_y. Consolidate back to single columns,
# preferring dataset6's values (borough_x, MM_x) as the primary source.
if "borough_x" in merged.columns:
    merged["borough"] = merged["borough_x"].fillna(merged["borough_y"])
    merged = merged.drop(columns=["borough_x", "borough_y"])
if "MM_x" in merged.columns:
    merged["MM"] = merged["MM_x"].fillna(merged["MM_y"])
    merged = merged.drop(columns=["MM_x", "MM_y"])

# --- Step 8: Sort so all readings for the same segment are grouped vertically ---
merged = merged.sort_values(
    by=["borough", "street", "fromSt", "toSt", "Yr", "M", "D", "HH", "MM"]
)
merged = merged.reset_index(drop=True)

print("Shape:", merged.shape)
print(merged.head())
print("\nNull counts:\n", merged.isnull().sum())
print("\nSample match distances (degrees):\n", dataset6["match_distance_deg"].describe())

merged.to_csv("merged_dataset_detailed.csv", index=False)
