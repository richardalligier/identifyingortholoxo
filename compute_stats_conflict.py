import argparse
import altair as alt
import pandas as pd
import datetime
from dataclasses import dataclass
from compute_stats_utils import *

parser = argparse.ArgumentParser(
    description='compute stats',
)
parser.add_argument('-detectedother',type=str)
parser.add_argument('-detectedref',type=str)
parser.add_argument('-conflict',type=str)
parser.add_argument('-folderfigures',type=str)
parser.add_argument('-foldertrajs',type=str)
parser.add_argument('-timesplit',type=int)
parser.add_argument('-r',type=float)
parser.add_argument('-dolmax',type=float)
args = parser.parse_args()

length_thresh = 30
altitude_thresh=20000
trajs=pd.read_parquet(f"{args.foldertrajs}").query("altitude>@altitude_thresh")
trajs["date"]=trajs["timestamp"].dt.date
isok = (trajs["timestamp"].diff().dt.total_seconds().to_numpy()>args.timesplit)
trajs["splitted"]= isok.cumsum()
tstart = trajs.groupby(["date","splitted","callsign","icao24"]).timestamp.min()
tend = trajs.groupby(["date","splitted","callsign","icao24"]).timestamp.max()
totalduration=(tend-tstart).sum()
savenumber(f"{totalduration}", f"{args.folderfigures}/durationall")
@dataclass
class Args:
    dolmax: float
    r: float


other = read_detected(args.detectedother)



d = {k: isole_altitude_dataset(v).query("length>@length_thresh").query("altitude_start>=@altitude_thresh") for k, v in {PROJ: other}.items()}
d[PROJORTHO] = extract_ortho(d[PROJ])
d[PROJLOXO] = extract_loxo(d[PROJ])
d[PROJORTHONOTLOXO] = extract_ortho_not_loxo(d[PROJ], Args(dolmax=args.dolmax, r=args.r))
d[PROJLOXONOTORTHO] = extract_loxo_not_ortho(d[PROJ], Args(dolmax=args.dolmax, r=args.r))
#add_intersection(d[PROJLOXONOTORTHO], d[PROJORTHONOTLOXO], suffix=PROJORTHONOTLOXO)
#add_intersection(d[PROJORTHONOTLOXO], d[PROJLOXONOTORTHO], suffix=PROJLOXONOTORTHO)

d[CONFLICT] = read_detected(args.conflict
    #"outfiles/detectedref.parquet"
    ).query("length>@length_thresh").query("altitude>=@altitude_thresh")
duration={}
for k in [PROJORTHONOTLOXO,CONFLICT,PROJLOXONOTORTHO]:
    duration[k]=datetime.timedelta(seconds=int((d[k].stop-d[k].start).sum()))
    savenumber(f"{duration[k]}", f"{args.folderfigures}/duration{k}")
    savenumber(f"{int(duration[k].total_seconds()/totalduration.total_seconds()*100)}",f"{args.folderfigures}/ratio{k}all")

#d={k:v for k,v in d.items()}
add_intersection(d[PROJLOXONOTORTHO], d[CONFLICT], suffix=CONFLICT)
add_intersection(d[PROJORTHONOTLOXO], d[CONFLICT], suffix=CONFLICT)

#add_intersection(d[CONFLICT],d[PROJLOXONOTORTHO],  suffix=PROJLOXONOTORTHO)
#add_intersection( d[CONFLICT],d[PROJORTHONOTLOXO],suffix=PROJORTHONOTLOXO)

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
for k, v in d.items():
    savenumber(f"{v.shape[0]}", f"{args.folderfigures}/card{k}")

inconflict={}
for k in [PROJLOXONOTORTHO,PROJORTHONOTLOXO]:
    df = d[k]
    #af = df.query(f"inclusion_ratio{CONFLICT}>0.9")
    inconflict[k]=((df.stop-df.start)*df[f"inclusion_ratio{CONFLICT}"]).sum()
    ratio=datetime.timedelta(seconds=int(inconflict[k]))
    savenumber(f"{ratio}", f"{args.folderfigures}/{k}inconflict")
    savenumber(f"{int(inconflict[k]/duration[k].total_seconds()*100)}",f"{args.folderfigures}/ratioconflict{k}")



