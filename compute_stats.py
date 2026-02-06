# %%
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from compute_stats_utils import *

parser = argparse.ArgumentParser(
    description='compute stats',
)
parser.add_argument('-detectedother',type=str)
parser.add_argument('-detectedref',type=str)
parser.add_argument('-conflict',type=str)
parser.add_argument('-folderfigures',type=str)
parser.add_argument('-r',type=float)
parser.add_argument('-dolmax',type=float)
args = parser.parse_args()



# %%

other = read_detected(args.detectedother
    #"outfiles/detected_alpha_mean_slope_3600_0.01_0.5_0.1_0.001_0.1_0"
)
ref = read_detected(args.detectedref
    #"outfiles/detectedref_alpha_mean_slope_3600_0.01_1_200_0.1_0.1"
)  # .query("lever<0.01")
d = {k: isole_altitude_dataset(v) for k, v in {BASELINE: ref, PROJ: other}.items()}
d[PROJORTHO] = extract_ortho(d[PROJ])
d[PROJLOXO] = extract_loxo(d[PROJ])
add_intersection(d[BASELINE], d[PROJORTHO], suffix=PROJORTHO)
add_intersection(d[PROJORTHO], d[BASELINE], suffix=BASELINE)

savenumber(f"{d[BASELINE]["iou"+PROJORTHO].mean():.5f}",f"{args.folderfigures}/maxioubaseline")
savenumber(f"{d[PROJORTHO]["iou"+BASELINE].mean():.5f}",f"{args.folderfigures}/maxiouprojortho")

# %%
import altair as alt

alt.data_transformers.enable("default", max_rows=None)
df = pd.concat(
    [
        d[BASELINE].assign(type=BASELINE),
        d[PROJORTHO].assign(type=PROJORTHO),
    ]
)

# %%

# Vega expressions: log10, pow
expr_log10 = "log(datum.domax) / log(10)"
expr_pow10 = "pow(10, datum._bin_log_start)"
expr_pow10_end = "pow(10, datum._bin_log_end)"

chart = (
    alt.Chart(df[["type", "domax"]])
    .transform_calculate(_log="{}".format(expr_log10))
    .transform_bin(
        as_=["_bin_log_start", "_bin_log_end"],
        field="_log",
        bin=alt.Bin(maxbins=50),
    )
    .transform_calculate(
        bin_left="{}".format(expr_pow10),
        bin_right="{}".format(expr_pow10_end),
    )
    .mark_bar(stroke="white", strokeWidth=2)
    .encode(
        x=alt.X("bin_left:Q")
        .scale(type="log", domain=[1, 2000])
        .axis(format="~s", grid=False)
        .title("Maximum distance (in m) between orthodromy and trajectory →"),
        x2="bin_right:Q",
        y=alt.Y("count():Q").title(None),
        color=alt.Color("type:N").legend(None),
        row=alt.Row("type:N").title(None),
        tooltip=[
            alt.Tooltip("bin_left:Q", title="left"),
            alt.Tooltip("bin_right:Q", title="right"),
            alt.Tooltip("count():Q"),
            alt.Tooltip("label:N"),
        ],
    )
    .properties(height=150, width=500)
    .configure_axis(
        labelFontSize=15,
        titleFontSize=16,
        labelFont="Roboto Condensed",
        titleFont="Roboto Condensed",
        titleAnchor="end",
    )
    .configure_header(
        labelFontSize=15,
        labelFont="Roboto Condensed",
        labelFontWeight="bold",
        labelAnchor="start",
        labelOrient="top",
    )
)
chart.save(f"{args.folderfigures}/domaxdist_alt.pdf")
chart

# %%

expr_log10 = "log(datum.length_min) / log(10)"

chart = (
    alt.Chart(df[["type", "length_min"]])
    .transform_calculate(_log="{}".format(expr_log10))
    .transform_bin(
        as_=["_bin_log_start", "_bin_log_end"],
        field="_log",
        bin=alt.Bin(maxbins=50),
    )
    .transform_calculate(
        bin_left="{}".format(expr_pow10),
        bin_right="{}".format(expr_pow10_end),
    )
    .mark_bar(stroke="white", strokeWidth=2)
    .encode(
        x=alt.X("bin_left:Q")
        .scale(type="log", domain=[0.1, 50])
        .axis(format="", grid=False)
        .title("Segment duration (in minutes) →"),
        x2="bin_right:Q",
        y=alt.Y("count():Q").title(None),
        color=alt.Color("type:N").legend(None),
        row=alt.Row("type:N").title(None),
        tooltip=[
            alt.Tooltip("bin_left:Q", title="left"),
            alt.Tooltip("bin_right:Q", title="right"),
            alt.Tooltip("count():Q"),
            alt.Tooltip("label:N"),
        ],
    )
    .properties(height=150, width=500)
    .configure_axis(
        labelFontSize=15,
        titleFontSize=16,
        labelFont="Roboto Condensed",
        titleFont="Roboto Condensed",
        titleAnchor="end",
    )
    .configure_header(
        labelFontSize=15,
        labelFont="Roboto Condensed",
        labelFontWeight="bold",
        labelAnchor="start",
        labelOrient="top",
    )
)
chart.save(f"{args.folderfigures}/lengthdist_alt.pdf")
chart

# %%

chart = alt.vconcat(
    alt.layer(
        base := alt.Chart(
            df[["type", f"iou{PROJORTHO}", f"iou{BASELINE}", "length"]]#.query("length>300")
        )
        .mark_bar(opacity=0.3)
        .encode(
            alt.X(f"iou{PROJORTHO}:Q")
            .bin(maxbins=30)
            .title(f"MaxIoU(x ∈ {BASELINE}, {PROJORTHO}) →"),
            alt.Y("count():Q").title(None),
            color=alt.Color("type:N").legend(None),
        )
        .properties(height=200, width=500),
        base.transform_filter(alt.datum.length > 300).mark_bar(opacity=1),
    ),
    alt.layer(
        base.encode(
            alt.X(f"iou{BASELINE}:Q")
            .bin(maxbins=30)
            .title(f"MaxIoU(x ∈ {PROJORTHO}, {BASELINE}) →")
        ),
        base.encode(alt.X(f"iou{BASELINE}:Q").bin(maxbins=30))
        .transform_filter(alt.datum.length > 300)
        .mark_bar(opacity=0.3),
    ),
).configure_axis(
    labelFontSize=15,
    titleFontSize=17,
    titleAnchor="end",
    labelFont="Roboto Condensed",
    titleFont="Roboto Condensed",
)
chart.save(f"{args.folderfigures}/maxiouprojorthobaseline_alt.pdf")
chart

# %%
add_intersection(d[PROJLOXO], d[PROJORTHO], suffix=PROJORTHO)

# %%

alt.Chart(df[["track_start"]]).mark_bar().encode(
    alt.X("track_start:Q", bin=alt.Bin(maxbins=36), title="Track start angle (°) →"),
    alt.Y("iou:Q", title=None),
).properties(height=300, width=500).configure_axis(
    labelFontSize=15,
    titleFontSize=17,
    titleAnchor="end",
    labelFont="Roboto Condensed",
    titleFont="Roboto Condensed",
)
# %%


# Polar histogram with Altair (using bar chart with angle bins)
df = d[PROJLOXO].copy()
num_bins = 36
df["angle_bin"] = pd.cut(
    df["track_start"], bins=np.linspace(0, 360, num_bins + 1), include_lowest=True
)
angle_bin_centers = [
    interval.left + (interval.right - interval.left) / 2
    for interval in df["angle_bin"].cat.categories
]
df["angle_bin_center"] = df["angle_bin"].apply(
    lambda x: x.left + (x.right - x.left) / 2 if pd.notnull(x) else np.nan
)

import math

# Create the circular axis lines for the number of observations
axis_rings = (
    alt.Chart(pd.DataFrame({"ring": [0.1 * i for i in range(1, 10)]}))
    .mark_arc(stroke="lightgrey", fill=None)
    .encode(theta=alt.value(2 * math.pi), radius=alt.Radius("ring").stack(False))
)
axis_rings_labels = axis_rings.mark_text(
    color="grey", radiusOffset=5, align="left", font="Roboto Condensed", fontSize=14
).encode(text="ring", theta=alt.value(math.pi / 4))

# Create the straight axis lines for the time of the day
axis_lines = (
    alt.Chart(
        pd.DataFrame(
            {
                "radius": 1,
                "theta": math.pi / 2,
                "hour": [0, 90, 180, 270],
            }
        )
    )
    .mark_arc(stroke="lightgrey", fill=None)
    .encode(
        theta=alt.Theta("theta").stack(True),
        radius=alt.Radius("radius"),
        radius2=alt.datum(0),
    )
)
axis_lines_labels = axis_lines.mark_text(
    font="Roboto Condensed",
    fontSize=16,
    color="grey",
    radiusOffset=5,
    thetaOffset=-math.pi / 4,
    # These adjustments could be left out with a larger radius offset, but they make the label positioning a bit clearner
    align=alt.expr(
        'datum.hour == "270" ? "right" : datum.hour == "90" ? "left" : "center"'
    ),
    baseline=alt.expr(
        'datum.hour == "0" ? "bottom" : datum.hour == "180" ? "top" : "middle"'
    ),
).encode(text="hour")

polar_chart = (
    alt.Chart(df[["length", "angle_bin_center", f"iou{PROJORTHO}"]])
    .mark_arc(innerRadius=10, stroke="white")
    .encode(
        alt.Theta("angle_bin_center:O"),
        alt.Radius("mean_iou:Q"),
    )
    .properties(
        width=400,
        height=400,
        title="Polar distribution of track_start angles (loxodromy segments)",
    )
)


# polar_chart.save("figures/track_start_polar_hist_altair.pdf")
chart = alt.layer(
    polar_chart.transform_filter(alt.datum.length > 300)
    .transform_aggregate(
        mean_iou=f"mean(iou{PROJORTHO})",
        groupby=["angle_bin_center"],
    )
    .mark_arc(opacity=0.3),
    polar_chart.transform_aggregate(
        mean_iou=f"mean(iou{PROJORTHO})",
        groupby=["angle_bin_center"],
    ),
    axis_rings,
    axis_rings_labels,
    axis_lines,
    axis_lines_labels,
    title=[f"Average MaxIoU(x ∈ {PROJORTHO}, {PROJLOXO}) per track angle", ""],
).configure_title(font="Roboto Condensed", fontSize=18, anchor="middle")
chart.save(f"{args.folderfigures}/track_start_polar_hist.pdf")
chart

# %%

from dataclasses import dataclass


@dataclass
class Args:
    dolmax: float
    r: float


d[PROJORTHONOTLOXO] = extract_ortho_not_loxo(d[PROJ], Args(dolmax=args.dolmax, r=args.r))
d[PROJLOXONOTORTHO] = extract_loxo_not_ortho(d[PROJ], Args(dolmax=args.dolmax, r=args.r))
add_intersection(d[PROJLOXONOTORTHO], d[PROJORTHONOTLOXO], suffix=PROJORTHONOTLOXO)
add_intersection(d[PROJORTHONOTLOXO], d[PROJLOXONOTORTHO], suffix=PROJLOXONOTORTHO)

# %%
chart = (
    alt.vconcat(
        alt.Chart(d[PROJORTHONOTLOXO][[f"iou{PROJLOXONOTORTHO}"]])
        .mark_bar()
        .encode(
            alt.X(f"iou{PROJLOXONOTORTHO}", bin=alt.Bin(maxbins=30))
            .scale(domain=[0, 1])
            .title(f"MaxIoU(x ∈ {PROJORTHONOTLOXO}, {PROJLOXONOTORTHO})"),
            alt.Y("count():Q")
            .title(None)
            .scale(type="log", domain=[1, 10000])
            .axis(grid=False),
        )
        .properties(height=150, width=500),
        alt.Chart(d[PROJLOXONOTORTHO][[f"iou{PROJORTHONOTLOXO}"]])
        .mark_bar(color="#f58518")
        .encode(
            alt.X(f"iou{PROJORTHONOTLOXO}", bin=alt.Bin(maxbins=30))
            .scale(domain=[0, 1])
            .title(f"MaxIoU(x ∈ {PROJLOXONOTORTHO}, {PROJORTHONOTLOXO})"),
            alt.Y("count():Q")
            .title(None)
            .scale(type="log", domain=[1, 10000])
            .axis(grid=False),
        )
        .properties(height=150, width=500),
    )
    .configure_axis(
        labelFontSize=15,
        titleFontSize=17,
        titleAnchor="end",
        labelFont="Roboto Condensed",
        titleFont="Roboto Condensed",
    )
    .resolve_axis(y="shared")
)
chart.save(f"{args.folderfigures}/maxiouloxoorthoonly_alt.pdf")
chart
# %%
d[CONFLICT] = read_detected(args.conflict
    #"outfiles/detectedref.parquet"
    )

add_intersection(d[PROJLOXONOTORTHO], d[CONFLICT], suffix=CONFLICT)
add_intersection(d[PROJORTHONOTLOXO], d[CONFLICT], suffix=CONFLICT)

# %%
chart = alt.vconcat(
    alt.Chart(d[PROJORTHONOTLOXO][[f"inclusion_ratio{CONFLICT}"]])
    .mark_bar()
    .encode(
        alt.X(f"inclusion_ratio{CONFLICT}", bin=alt.Bin(maxbins=30)).title(
            f"MaxIoL(x ∈ {PROJORTHONOTLOXO}, {CONFLICT})"
        ),
        alt.Y("count():Q")
        .title(None)
        .scale(type="log", domain=[1, 10000])
        .axis(grid=False),
    )
    .properties(height=150, width=500),
    alt.Chart(d[PROJLOXONOTORTHO][[f"inclusion_ratio{CONFLICT}"]])
    .mark_bar(color="#f58518")
    .encode(
        alt.X(f"inclusion_ratio{CONFLICT}", bin=alt.Bin(maxbins=30)).title(
            f"MaxIoL(x ∈ {PROJLOXONOTORTHO}, {CONFLICT})"
        ),
        alt.Y("count():Q")
        .title(None)
        .scale(type="log", domain=[1, 10000])
        .axis(grid=False),
    )
    .properties(height=150, width=500),
).configure_axis(
    labelFontSize=15,
    titleFontSize=17,
    titleAnchor="end",
    labelFont="Roboto Condensed",
    titleFont="Roboto Condensed",
)
chart.save(f"{args.folderfigures}/maxiolortho_loxo_alt.pdf")
chart
for k, v in d.items():
    savenumber(f"{v.shape[0]}", f"{args.folderfigures}/card{k}")

