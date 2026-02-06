# %%

import json
from typing import Any, Dict, Optional, cast

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.crs import PlateCarree
from cartopy.mpl.geoaxes import GeoAxes
from traffic.core import Flight

import pandas as pd


def _to_flight(points) -> Flight:
    df = pd.DataFrame(points)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    if "track_angle" in df.columns:
        df = df.rename(columns={"track_angle": "track"})
    return Flight(df).resample("1s")


def plot_closest_conflict(
    sample: Dict[str, Any],
    save_path: Optional[str] = None,
    include_flight_plan: bool = False,  # set True if you still want navaids
    pick_by: str = "predicted",         # "predicted" or "actual"
    segments_df: Optional[pd.DataFrame] = None,
) -> None:
    # --- projection / axes
    lambert = ccrs.LambertConformal(central_longitude=3.0, central_latitude=46.5)
    fig, ax_ = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(projection=lambert))
    ax = cast(GeoAxes, ax_)
    ax.spines["geo"].set_visible(False)

    # --- palette
    # c_actual = "#2E77BB"   # deviated actual
    c_actual = "black"
    c_devseg = "#D81B60"   # deviation segment
    c_pred   = "#7A7A7A"   # deviated predicted
    # c_other  = "#54a24b"   # neighbour
    c_other = "gray"
    c_ortho = "yellow"
    plane_sz = 320

    # ---------------- Deviated aircraft (actual + deviation segment)
    dev_id = sample["deviated_aircraft"]["flight_id"]
    f_actual = _to_flight(sample["trajectories"][dev_id]["points"])
    f_actual.plot(ax, color=c_actual, linewidth=2.2, label="ADS-B trajectory")

    start_dev = pd.to_datetime(sample["deviated_aircraft"]["start_deviation"], unit="s", utc=True)
    stop_dev  = pd.to_datetime(sample["deviated_aircraft"]["stop_deviation"],  unit="s", utc=True)
    # seg = f_actual.between(start_dev, stop_dev)
    # if seg is not None:
    #     seg.plot(ax, color=c_devseg, linewidth=2.8, label="Deviation segment")

    # ---------------- Plot deviation segments if provided
    seen = set()
    tab10 = plt.cm.tab10.colors
    # ortho_colors = [tab10[i] for i in (1, 0,2,3, 4)]
    # loxo_color = tab10[0]
    segments_df = segments_df.astype({"segment_number": "int"})
    if segments_df is not None:
        print(segments_df)
        max_loxo = segments_df["maxloxo"].iloc[0]#query("identified=='loxodromy'").segment_number.max()
        for _, row in segments_df.iterrows():
            seg = f_actual.between(row["start"], row["stop"])
            print(row['segment_number'])
            icolor =row['segment_number']-1
            print(icolor)
            if seg is not None:
                if row['identified'] == "loxodromy":
                    label = f"Loxodromy #{row['segment_number']}"
                    seg.plot(ax, color=tab10[icolor], linewidth=2.8, label=label, zorder=9)
                    seen.add(f"Loxodromy #{row['segment_number']}")
                else:
                    icolor+=max_loxo
                    # label = "Orthodromy" if "Orthodromy" not in seen else "_nolegend_"
                    label = f"Orthodromy #{row['segment_number']}"
                    seg.plot(ax, color=tab10[icolor], linewidth=2.8, label=label, zorder=9, alpha=1)
                    seen.add("Orthodromy")
            else:
                print(row["segment_number"])


    # ---------------- Deviated predicted
    f_pred = _to_flight(sample["deviated_aircraft"]["predicted_trajectory"]["points"])
    f_pred.plot(ax, linestyle="--", color=c_pred, linewidth=2.0, label="Deviated (predicted)", alpha=0.5)

    # ---------------- Choose closest neighbour (by predicted/actual CPA distance)
    pred_pairs = sample.get("predicted_pairwise", {}) or {}
    act_pairs  = sample.get("actual_pairwise", {}) or {}

    candidates = []
    for fid, c in pred_pairs.items():
        if fid not in sample.get("trajectories", {}):
            continue
        pred_nm = c.get("lateral_dist_at_tcpa")
        act_nm  = (act_pairs.get(fid) or {}).get("lateral_dist_at_tcpa")
        t_cpa   = c.get("time_at_cpa")
        if pred_nm is None or t_cpa is None:
            continue
        candidates.append((fid, float(pred_nm), (None if act_nm is None else float(act_nm)), int(t_cpa)))

    if not candidates:
        # nothing else to plot; still show deviated flight cleanly
        _finalize(ax, save_path)
        return

    if pick_by == "actual":
        # fall back to predicted if actual is missing
        fid, pred_nm, act_nm, t_cpa = min(
            candidates, key=lambda t: t[2] if t[2] is not None else float("inf")
        )
        if act_nm is None:
            fid, pred_nm, act_nm, t_cpa = min(candidates, key=lambda t: t[1])
    else:
        fid, pred_nm, act_nm, t_cpa = min(candidates, key=lambda t: t[1])

    # ---------------- Plot the neighbour
    f_other = _to_flight(sample["trajectories"][fid]["points"])
    f_other.plot(ax, color=c_other, linewidth=2.2, label="Closest neighbour")

    # ---------------- Mark CPA on all trajectories
    cpa_time = pd.to_datetime(t_cpa, unit="s", utc=True)
    f_other.at(cpa_time).plot(ax, color=c_other, s=plane_sz,
                              text_kw={"s": "", "color": c_other}, zorder=10, alpha=1)
    f_actual.at(cpa_time).plot(ax, color=c_actual, s=plane_sz,
                               text_kw={"s": ""}, zorder=10)
    f_pred.at(cpa_time).plot(ax, color=c_pred, s=plane_sz,
                             text_kw={"s": ""}, zorder=10, alpha=0.5)

    # ---------------- Optional: flight plan navaids
    if include_flight_plan and "flight_plan" in sample.get("deviated_aircraft", {}):
        first = True
        for nav in sample["deviated_aircraft"]["flight_plan"][8:15]:
            ax.scatter(
                nav["longitude"], nav["latitude"],
                marker="x", s=70, color="#ff9d98",
                transform=PlateCarree(), zorder=12,
                label="Flight plan navaid" if first else None
            )
            first = False

    # ---------------- Annotation (top-left)
    lines = ["Lateral distance at CPA",
             f"Predicted: {pred_nm:.1f} NM"]
    if act_nm is not None:
        lines.append(f"Actual: {act_nm:.1f} NM")

    ax.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax.transAxes, va="top",
        fontsize=12.5,
        bbox=dict(facecolor="white", edgecolor="0.8", boxstyle="round,pad=0.35")
    )

    # ---------------- Legend + export
    leg = ax.legend(loc="lower left", fontsize=11, frameon=True)
    leg.set_zorder(100)
    _finalize(ax, save_path)


def _finalize(ax: GeoAxes, save_path: Optional[str]) -> None:
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(ax.figure)

def main():
    import figures
    import argparse
    import matplotlib.pyplot as plt
    # config = read_config()
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-folderfigures')
    args = parser.parse_args()
    config = figures.read_config()
    print(config)
    with open(f"{config.FOLDER_DETECTEDREF_JSON}/39680815_1659854804_1659855207.json", "r", encoding="utf-8") as f:
        sample = json.load(f)
    print(sample)
    segments =pd.read_csv(f"{args.folderfigures}/withpurelongest.csv")
    #.query("pure")
    segments = segments.rename(columns={"segment number": "segment_number", "type": "identified"})
    segments['start'] = pd.to_datetime(segments["start"], unit="s", utc=True)
    segments['stop'] = pd.to_datetime(segments["stop"], unit="s", utc=True)
    plot_closest_conflict(sample, save_path=f"{args.folderfigures}/conflictsituation.pdf", include_flight_plan=True, pick_by="predicted", segments_df=segments)

# %%
main()
