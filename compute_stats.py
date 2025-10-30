# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

COUNT = "count [-]"
PROJ = "SegmentsProj"
PROJORTHO = PROJ + "Ortho"
PROJLOXO = PROJ + "Loxo"
PROJORTHONOTLOXO = PROJORTHO + "NotLoxo"
PROJLOXONOTORTHO = PROJLOXO + "NotOrtho"
BASELINE = "SegmentsBaseline"
CONFLICT = "SegmentsDeconfliction"


def read_detected(fname):
    df = pd.read_parquet(fname)
    if "date" not in df:
        df["date"] = df["start"].astype("datetime64[s]").dt.date
    df["length"] = df["stop"] - df["start"]
    df["length_min"] = df["length"] / 60
    df["datetime_start"] = df["start"].astype("datetime64[s]")  # .dt.date
    df["datetime_stop"] = df["stop"].astype("datetime64[s]")  # .dt.date
    return df.sort_values(["icao24", "start"])


def extract_loxo_not_ortho(df, args):
    return (
        df.query("iswhat=='loxodromy'")
        .query("dolmax>=@args.dolmax")
        .query("dlmax<@args.r*domax")
        .query("dlmax<@args.r*dolmax")
    )


def extract_ortho_not_loxo(df, args):
    return (
        df.query("iswhat=='orthodromy'")
        .query("dolmax>=@args.dolmax")
        .query("domax<@args.r*dlmax")
        .query("domax<@args.r*dolmax")
    )


def extract_ortho(df):
    return df.query(
        "iswhat=='orthodromy'"
    )  # .query("domax<100")#.query("domax<@args.r*dlmax")#.query("npts>10")#.query("dolmax>20")


def extract_loxo(df):
    return df.query(
        "iswhat=='loxodromy'"
    )  # .query("domax<100")#.query("domax<@args.r*dlmax")#.query("npts>10")#.query("dolmax>20")


def isole_altitude_dataset(df):
    return (
        df.query("altitude_start>=20000")
        .query("altitude_stop>=20000")
        .query("abs(altitude_stop-altitude_start)<200")
        .query("length>30")
    )


def intersection(l, q):
    start = max(q.start, l.start)
    end = min(l.stop, q.stop)
    return max(end - start, 0.0)


def union(l, q):
    start = min(q.start, l.start)
    end = max(l.stop, q.stop)
    return max(end - start, 0.0)


def is_included(l, q):
    return q.start <= l.start and l.stop <= q.stop


def inclusion_ratio(l, q):
    if is_included(l, q):
        return 1
    else:
        return intersection(l, q) / (l.stop - l.start)  # /union(l,q)


# def inclusion(l,q):# l C q ???
#     if is_included(l,q):
#         return 1
#     else:
#         inter=intersection(l,q)
#         if inter==0:
#             return 0
#         else:
#             return inter/(l.stop-l.start)


def getkey(line):
    return (line.icao24, line.start, line.stop)


def map_key(d, f):
    res = {}
    for k, v in d.items():
        res[f[k]] = v
    return res


def add_intersection(af, cf, suffix=""):
    res = {}
    d = {k: k + suffix for k in ["iou", "inclusion_ratio", "inclusion"]}
    af[d["iou"]] = 0.0
    af[d["inclusion_ratio"]] = 0.0
    # af[d["inclusion"]]=0.
    for i, line in tqdm.tqdm(cf.iterrows()):
        k = getkey(line)
        res[k] = []
        qf = af.query("date==@line.date").query("icao24==@line.icao24")
        for _, qline in qf.iterrows():
            leninter = intersection(qline, line)
            if leninter > 0.0:
                af.loc[qline.name, d["iou"]] = max(
                    leninter / union(qline, line), af.loc[qline.name, d["iou"]]
                )
                af.loc[qline.name, d["inclusion_ratio"]] = max(
                    inclusion_ratio(qline, line),
                    af.loc[qline.name, d["inclusion_ratio"]],
                )
                # af.loc[qline.name,d["inclusion"]]=max(inclusion(qline,line),af.loc[qline.name,d["inclusion"]])
                res[k].append(qline)
    return res


def plothist(d_ortho, vstr, ystr, bins=50, semilog=False):
    if semilog:
        bins = np.geomspace(
            min(v[vstr].min() for v in d_ortho.values()),
            max(v[vstr].max() for v in d_ortho.values()),
            bins + 1,
        )
    if isinstance(vstr, str):
        plt.hist(tuple(v[vstr] for k, v in d_ortho.items()), bins=bins)
    else:
        plt.hist(tuple(v[vstr[k]] for k, v in d_ortho.items()), bins=bins)
    if semilog:
        plt.xscale("log")
        ystr += " (log scale)"
    plt.xlabel(ystr)
    plt.ylabel(COUNT)
    plt.gca().legend(list(d_ortho.keys()))


def savefig(fig, fname, width=4):
    fig.set_tight_layout({"pad": 0})
    fig.set_figwidth(width)
    plt.savefig(f"{fname}", dpi=300, bbox_inches="tight")
    plt.clf()


def savenumber(s, fname):
    with open(fname + ".tex", "w") as f:
        f.write(s)


# %%
other = read_detected(
    "outfiles/detected_alpha_mean_slope_3600_0.01_0.5_0.1_0.001_0.1_0"
)
ref = read_detected(
    "outfiles/detectedref_alpha_mean_slope_3600_0.01_1_200_0.1_0.1"
)  # .query("lever<0.01")
d = {k: isole_altitude_dataset(v) for k, v in {BASELINE: ref, PROJ: other}.items()}
d[PROJORTHO] = extract_ortho(d[PROJ])
d[PROJLOXO] = extract_loxo(d[PROJ])
add_intersection(d[BASELINE], d[PROJORTHO], suffix=PROJORTHO)
add_intersection(d[PROJORTHO], d[BASELINE], suffix=BASELINE)

# %%
import altair as alt

alt.data_transformers.enable("default", max_rows=None)
df = pd.concat(
    [
        d["SegmentsBaseline"].assign(type="Baseline"),
        d["SegmentsProjOrtho"].assign(type="Orthodromy"),
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
chart.save("figures/domaxdist_alt.pdf")
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
chart.save("figures/lengthdist_alt.pdf")
chart

# %%

chart = alt.vconcat(
    alt.layer(
        base := alt.Chart(
            df[["type", "iouSegmentsProjOrtho", "iouSegmentsBaseline", "length"]]
        )
        .mark_bar(opacity=0.3)
        .encode(
            alt.X("iouSegmentsProjOrtho:Q")
            .bin(maxbins=30)
            .title("MaxIoU(x ∈ Baseline, Orthodromy) →"),
            alt.Y("count():Q").title(None),
            color=alt.Color("type:N").legend(None),
        )
        .properties(height=200, width=500),
        base.transform_filter(alt.datum.length > 300).mark_bar(opacity=1),
    ),
    alt.layer(
        base.encode(
            alt.X("iouSegmentsBaseline:Q")
            .bin(maxbins=30)
            .title("MaxIoU(x ∈ Orthodromy, Baseline) →")
        ),
        base.encode(alt.X("iouSegmentsBaseline:Q").bin(maxbins=30))
        .transform_filter(alt.datum.length > 300)
        .mark_bar(opacity=1),
    ),
).configure_axis(
    labelFontSize=15,
    titleFontSize=17,
    titleAnchor="end",
    labelFont="Roboto Condensed",
    titleFont="Roboto Condensed",
)
chart.save("figures/maxiouprojorthobaseline_alt.pdf")
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
    title=["Average MaxIoU(x ∈ Orthodromy, Loxodromy) per track angle", ""],
).configure_title(font="Roboto Condensed", fontSize=18, anchor="middle")
chart.save("figures/track_start_polar_hist.pdf")
chart

# %%

from dataclasses import dataclass


@dataclass
class Args:
    dolmax: float
    r: float


d[PROJORTHONOTLOXO] = extract_ortho_not_loxo(d[PROJ], Args(dolmax=30, r=0.5))
d[PROJLOXONOTORTHO] = extract_loxo_not_ortho(d[PROJ], Args(dolmax=30, r=0.5))
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
            .title("MaxIoU(x ∈ OrthodromyNotLoxo, LoxodromyNotOrtho)"),
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
            .title("MaxIoU(x ∈ LoxodromyNotOrtho, OrthodromyNotLoxo)"),
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
chart.save("figures/maxiouloxoorthoonly_alt.pdf")
chart
# %%
d[CONFLICT] = read_detected("outfiles/detectedref.parquet")

add_intersection(d[PROJLOXONOTORTHO], d[CONFLICT], suffix=CONFLICT)
add_intersection(d[PROJORTHONOTLOXO], d[CONFLICT], suffix=CONFLICT)

# %%
chart = alt.vconcat(
    alt.Chart(d[PROJORTHONOTLOXO][[f"inclusion_ratio{CONFLICT}"]])
    .mark_bar()
    .encode(
        alt.X(f"inclusion_ratio{CONFLICT}", bin=alt.Bin(maxbins=30)).title(
            "MaxIoL(x ∈ OrthodromyNotLoxo, Deconfliction)"
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
            "MaxIoL(x ∈ LoxodromyNotOrtho, Deconfliction)"
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
chart.save("figures/maxiolortho_loxo_alt.pdf")
chart


# %%
def main():
    import argparse

    import matplotlib.pyplot as plt

    # config = read_config()
    parser = argparse.ArgumentParser(
        description="fit trajectories and save them in folders",
    )
    parser.add_argument("-detectedref")
    parser.add_argument("-detectedother")
    parser.add_argument("-conflict")
    parser.add_argument("-folderfigures")
    parser.add_argument("-r", type=float, required=True)
    parser.add_argument("-dolmax", type=float, required=True)
    args = parser.parse_args()
    ref = read_detected(args.detectedref)
    other = read_detected(args.detectedother)  # .query("lever<0.01")
    print(f"{ref.shape=}")
    print(f"{other.shape=}")
    print(f"{other.query("iswhat=='orthodromy'").shape=}")
    print(f"{other.altitude_start.describe()=}")
    print(f"{other.query("iswhat=='orthodromy'").altitude_start.describe()=}")
    print(f"{other.query("iswhat=='loxodromy'").altitude_start.describe()=}")
    d = {k: isole_altitude_dataset(v) for k, v in {BASELINE: ref, PROJ: other}.items()}
    d[PROJORTHO] = extract_ortho(d[PROJ])
    d[PROJLOXO] = extract_loxo(d[PROJ])
    d[PROJORTHONOTLOXO] = extract_ortho_not_loxo(d[PROJ], args)
    d[PROJLOXONOTORTHO] = extract_loxo_not_ortho(d[PROJ], args)
    d[CONFLICT] = read_detected(args.conflict)
    del d[PROJ]
    for k, v in d.items():
        savenumber(f"{v.shape[0]}", f"{args.folderfigures}/card{k}")
    add_intersection(d[BASELINE], d[PROJORTHO], suffix=PROJORTHO)
    add_intersection(d[PROJORTHO], d[BASELINE], suffix=BASELINE)
    fig = plt.figure()
    plothist(
        {k: d[k] for k in [PROJORTHO, BASELINE]},
        "domax",
        "maximum distance between orthodromy and the trajectory\n on considered segment [m]",
        semilog=True,
    )
    savefig(fig, f"{args.folderfigures}/domaxdist.pdf", width=6)
    fig = plt.figure()
    plothist(
        {k: d[k] for k in [PROJORTHO, BASELINE]},
        "length_min",
        "segment duration [min]",
        semilog=True,
    )
    savefig(fig, f"{args.folderfigures}/lengthdist.pdf", width=6)
    fig = plt.figure()
    f = {
        PROJORTHO: f"distribution of MaxIoU(x,{BASELINE}) \nfor x in {PROJORTHO}",
        BASELINE: f"distribution of MaxIoU(x,{PROJORTHO}) \nfor x in {BASELINE}",
    }
    iou = {
        PROJORTHO: BASELINE,
        BASELINE: PROJORTHO,
    }
    plothist(
        {f[k]: d[k] for k in [PROJORTHO, BASELINE]},
        {f[k]: f"iou{iou[k]}" for k in [PROJORTHO, BASELINE]},
        "distribution of MaxIoU(x,SegmentsSet) [-]",
    )
    savefig(fig, f"{args.folderfigures}/maxiouprojorthobaseline.pdf")
    plothist(
        {f[k]: d[k].query("length>300") for k in [PROJORTHO, BASELINE]},
        {f[k]: f"iou{iou[k]}" for k in [PROJORTHO, BASELINE]},
        "distribution of MaxIoU(x,SegmentsSet) [-]",
    )
    savefig(fig, f"{args.folderfigures}/maxiouprojorthobaseline300.pdf")
    add_intersection(d[PROJLOXO], d[PROJORTHO], suffix=PROJORTHO)
    # add_intersection(d[PROJORTHO],d[BASELINE],suffix=BASELINE)
    fig = plt.figure()
    df = d[PROJLOXO]
    res = df.groupby(pd.cut(df.track_start, bins=36))[f"iou{PROJORTHO}"].mean()
    n = res.index.categories.left.shape[0]
    x = np.zeros(1 + n)
    x[:-1] = res.index.categories.left
    x[-1] = res.index.categories.right[-1]
    plt.stairs(res.values, x, fill=True)
    plt.xlabel("$\\text{track} [^\\circ]$")
    ystr = f"MaxIoU(x,{PROJORTHO}) for x in {PROJLOXO}\n averaged in each track bin"
    plt.ylabel(ystr)
    savefig(fig, f"{args.folderfigures}/maxiouortholoxotrack.pdf")
    fig = plt.figure()
    df = d[PROJLOXO]
    res = df.groupby(pd.cut(df.dolmax, bins=15))[f"iou{PROJORTHO}"].mean()
    n = res.index.categories.left.shape[0]
    x = np.zeros(1 + n)
    x[:-1] = res.index.categories.left
    x[-1] = res.index.categories.right[-1]
    plt.stairs(res.values, x, fill=True)
    plt.xlabel(
        "maximum distance between orthodromy and loxodromy\n on considered segment [m]"
    )
    ystr = f"MaxIoU(x,{PROJORTHO}) for x in {PROJLOXO}\n averaged in each distance bin"
    plt.ylabel(ystr)
    savefig(fig, f"{args.folderfigures}/maxiouortholoxodolmax.pdf")
    add_intersection(d[PROJLOXONOTORTHO], d[CONFLICT], suffix=CONFLICT)
    add_intersection(d[PROJORTHONOTLOXO], d[CONFLICT], suffix=CONFLICT)
    fig = plt.figure()
    plt.hist(d[PROJORTHONOTLOXO][f"inclusion_ratio{CONFLICT}"])
    plt.xlabel(f"MaxIoL(x,{CONFLICT}) \nfor x in {PROJORTHONOTLOXO}")
    plt.ylabel(COUNT)
    savefig(fig, f"{args.folderfigures}/maxiolortho.pdf")
    fig = plt.figure()
    plt.hist(d[PROJLOXONOTORTHO][f"inclusion_ratio{CONFLICT}"])
    plt.xlabel(f"MaxIoL(x,{CONFLICT}) \nfor x in {PROJLOXONOTORTHO}")
    plt.ylabel(COUNT)
    savefig(fig, f"{args.folderfigures}/maxiolloxo.pdf")

    add_intersection(d[PROJLOXONOTORTHO], d[PROJORTHONOTLOXO], suffix=PROJORTHONOTLOXO)
    add_intersection(d[PROJORTHONOTLOXO], d[PROJLOXONOTORTHO], suffix=PROJLOXONOTORTHO)
    fig = plt.figure()
    f = {
        PROJORTHONOTLOXO: f"distribution of MaxIoU(x,{PROJLOXONOTORTHO}) \nfor x in {PROJORTHONOTLOXO}",
        PROJLOXONOTORTHO: f"distribution of MaxIoU(x,{PROJORTHONOTLOXO}) \nfor x in {PROJLOXONOTORTHO}",
    }
    iou = {
        PROJLOXONOTORTHO: PROJORTHONOTLOXO,
        PROJORTHONOTLOXO: PROJLOXONOTORTHO,
    }
    toiter = [PROJORTHONOTLOXO, PROJLOXONOTORTHO]
    plothist(
        {f[k]: d[k] for k in toiter},
        {f[k]: f"iou{iou[k]}" for k in toiter},
        "distribution of MaxIoU(x,SegmentsSet) [-]",
    )
    savefig(fig, f"{args.folderfigures}/maxiouloxoorthoonly.pdf")


# if __name__ == "__main__":
#     main()
