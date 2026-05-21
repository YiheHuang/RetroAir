"""
03_extract_osm.py —— 在城市坐标上提取 OSM 缓冲特征
输出: data/processed/osm_features.csv
用法: python scripts/03_extract_osm.py [--radius 1000]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd
import osmnx as ox
from config import DATA_PROCESSED, CITY_COORDS, OSM_POI_TAGS, OSM_RADIUS


def extract_buffer_features(lat: float, lon: float, radius: int) -> dict:
    """提取以(lat,lon)为中心、radius为半径的OSM特征"""
    features = {}
    for name, tags in OSM_POI_TAGS.items():
        try:
            gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius)
            features[f"poi_{name}_count"] = len(gdf)
        except Exception:
            features[f"poi_{name}_count"] = 0

    try:
        G = ox.graph_from_point((lat, lon), dist=radius, network_type="drive")
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        features["road_total_length_m"] = (
            float(edges["length"].sum()) if "length" in edges.columns else 0.0
        )
        features["road_segment_count"] = len(edges)
        if "highway" in edges.columns:
            for hw in ["motorway","trunk","primary","secondary","tertiary","residential"]:
                mask = edges["highway"].str.contains(hw, na=False)
                features[f"road_{hw}_length_m"] = (
                    float(edges.loc[mask, "length"].sum())
                    if "length" in edges.columns else 0.0
                )
        else:
            for hw in ["motorway","trunk","primary","secondary","tertiary","residential"]:
                features[f"road_{hw}_length_m"] = 0.0
    except Exception:
        features["road_total_length_m"] = 0.0
        features["road_segment_count"] = 0
        for hw in ["motorway","trunk","primary","secondary","tertiary","residential"]:
            features[f"road_{hw}_length_m"] = 0.0

    area_km2 = (3.14159 * radius ** 2) / 1e6
    features["road_density_km_per_km2"] = (
        features["road_total_length_m"] / 1000 / area_km2 if area_km2 > 0 else 0.0
    )
    total_poi = sum(features.get(f"poi_{n}_count", 0) for n in OSM_POI_TAGS)
    features["poi_total_density_per_km2"] = total_poi / area_km2 if area_km2 > 0 else 0.0
    return features


def main():
    parser = argparse.ArgumentParser(description="提取OSM缓冲特征")
    parser.add_argument("--radius", type=int, default=OSM_RADIUS)
    args = parser.parse_args()

    cities = list(CITY_COORDS.keys())
    print(f"提取 OSM 特征: {len(cities)} 城市, 半径 {args.radius}m")

    all_features = []
    for i, city in enumerate(cities):
        lat, lon = CITY_COORDS[city]
        print(f"[{i+1}/{len(cities)}] {city} ({lat:.4f}, {lon:.4f})...")
        try:
            feat = extract_buffer_features(lat, lon, args.radius)
            feat["city"] = city
            feat["lat"] = lat
            feat["lon"] = lon
            all_features.append(feat)
            print(f"  OK POI={feat['poi_total_density_per_km2']:.1f}/km2, "
                  f"路网={feat['road_density_km_per_km2']:.1f}km/km2")
        except Exception as e:
            print(f"  ERR: {str(e)[:120]}")

    if not all_features:
        print("错误: 未提取到任何OSM特征!")
        sys.exit(1)

    df = pd.DataFrame(all_features)
    out = DATA_PROCESSED / "osm_features.csv"
    df.to_csv(out, index=False)
    print(f"\n已保存: {out} ({df.shape[0]} 行 x {df.shape[1]} 列)")


if __name__ == "__main__":
    main()
