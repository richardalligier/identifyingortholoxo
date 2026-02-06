import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

COUNT = "count [-]"
PROJ = "Proj"
PROJORTHO = PROJ + "Ortho"
PROJLOXO = PROJ + "Loxo"
PROJORTHONOTLOXO = PROJORTHO + "NotLoxo"
PROJLOXONOTORTHO = PROJLOXO + "NotOrtho"
BASELINE = "Baseline"
CONFLICT = "Deconfliction"


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
    for _, line in tqdm.tqdm(cf.iterrows()):
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