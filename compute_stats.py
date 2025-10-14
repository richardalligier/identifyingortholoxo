import pandas as pd
import tqdm
import numpy as np
import matplotlib.pyplot as plt


COUNT="count [-]"
PROJ = "SegmentsProj"
PROJORTHO=PROJ+"Ortho"
PROJLOXO=PROJ+"Loxo"
PROJORTHONOTLOXO=PROJORTHO+"NotLoxo"
PROJLOXONOTORTHO=PROJLOXO+"NotOrtho"
BASELINE = "SegmentsBaseline"
CONFLICT = "SegmentsDeconfliction"

def read_detected(fname):
    df = pd.read_parquet(fname)
    if "date" not in df:
        df["date"]=df["start"].astype("datetime64[s]").dt.date
    df["length"]=df["stop"]-df["start"]
    df["datetime_start"] = df["start"].astype("datetime64[s]")#.dt.date
    df["datetime_stop"] = df["stop"].astype("datetime64[s]")#.dt.date
    return df.sort_values(["icao24","start"])


def extract_loxo_not_ortho(df,args):
    return df.query("iswhat=='loxodromy'").query("dolmax>=@args.dolmax").query("dlmax<@args.r*domax").query("dlmax<@args.r*dolmax")
def extract_ortho_not_loxo(df,args):
    return df.query("iswhat=='orthodromy'").query("dolmax>=@args.dolmax").query("domax<@args.r*dlmax").query("domax<@args.r*dolmax")

def extract_ortho(df):
    return df.query("iswhat=='orthodromy'")#.query("domax<100")#.query("domax<@args.r*dlmax")#.query("npts>10")#.query("dolmax>20")

def extract_loxo(df):
    return df.query("iswhat=='loxodromy'")#.query("domax<100")#.query("domax<@args.r*dlmax")#.query("npts>10")#.query("dolmax>20")

def isole_altitude_dataset(df):
    return df.query("altitude_start>=20000").query("altitude_stop>=20000").query("abs(altitude_stop-altitude_start)<200").query("length>30")

def intersection(l,q):
    start = max(q.start,l.start)
    end = min(l.stop,q.stop)
    return max(end-start,0.)

def union(l,q):
    start = min(q.start,l.start)
    end = max(l.stop,q.stop)
    return max(end-start,0.)

def is_included(l,q):
    return q.start<=l.start and l.stop <=q.stop

def inclusion_ratio(l,q):
    if is_included(l,q):
        return 1
    else:
        return intersection(l,q)/(l.stop-l.start)#/union(l,q)

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
    return (line.icao24,line.start,line.stop)

def map_key(d,f):
    res = {}
    for k,v in d.items():
        res[f[k]]=v
    return res

def add_intersection(af,cf,suffix=""):
    res = {}
    d={k:k+suffix for k in ["iou","inclusion_ratio","inclusion"]}
    af[d["iou"]]=0.
    af[d["inclusion_ratio"]]=0.
    # af[d["inclusion"]]=0.
    for i,line in tqdm.tqdm(cf.iterrows()):
        k = getkey(line)
        res[k]=[]
        qf = af.query("date==@line.date").query("icao24==@line.icao24")
        for _, qline in qf.iterrows():
            leninter=intersection(qline,line)
            if leninter>0.:
                af.loc[qline.name,d["iou"]]=max(leninter/union(qline,line),af.loc[qline.name,d["iou"]])
                af.loc[qline.name,d["inclusion_ratio"]]=max(inclusion_ratio(qline,line),af.loc[qline.name,d["inclusion_ratio"]])
                # af.loc[qline.name,d["inclusion"]]=max(inclusion(qline,line),af.loc[qline.name,d["inclusion"]])
                res[k].append(qline)
    return res

def plothist(d_ortho,vstr,ystr,bins=50):
    if isinstance(vstr,str):
        plt.hist(tuple(v[vstr] for k,v in d_ortho.items()),bins=bins)
    else:
        plt.hist(tuple(v[vstr[k]] for k,v in d_ortho.items()),bins=bins)
    plt.xlabel(ystr)
    plt.ylabel(COUNT)
    plt.gca().legend(list(d_ortho.keys()))

def savefig(fig,fname,width=4):
    fig.set_tight_layout({'pad':0})
    fig.set_figwidth(width)
    plt.savefig(f"{fname}", dpi=300, bbox_inches='tight')
    plt.clf()

def savenumber(s,fname):
    with open(fname+".tex",'w') as f:
        f.write(s)
def main():
    import argparse
    import matplotlib.pyplot as plt
    # config = read_config()
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-detectedref')
    parser.add_argument('-detectedother')
    parser.add_argument('-conflict')
    parser.add_argument('-folderfigures')
    parser.add_argument('-r',type=float,required=True)
    parser.add_argument('-dolmax',type=float,required=True)
    args = parser.parse_args()
    ref = read_detected(args.detectedref)
    other = read_detected(args.detectedother)#.query("lever<0.01")
    print(f"{ref.shape=}")
    print(f"{other.shape=}")
    print(f"{other.query("iswhat=='orthodromy'").shape=}")
    print(f"{other.altitude_start.describe()=}")
    print(f"{other.query("iswhat=='orthodromy'").altitude_start.describe()=}")
    print(f"{other.query("iswhat=='loxodromy'").altitude_start.describe()=}")
    d = {k:isole_altitude_dataset(v) for k,v in {BASELINE:ref,PROJ:other}.items()}
    d[PROJORTHO]=extract_ortho(d[PROJ])
    d[PROJLOXO]=extract_loxo(d[PROJ])
    d[PROJORTHONOTLOXO]=extract_ortho_not_loxo(d[PROJ],args)
    d[PROJLOXONOTORTHO]=extract_loxo_not_ortho(d[PROJ],args)
    d[CONFLICT]=read_detected(args.conflict)
    del d[PROJ]
    for k,v in d.items():
        savenumber(f"{v.shape[0]}",f"{args.folderfigures}/card{k}")
    add_intersection(d[BASELINE],d[PROJORTHO],suffix=PROJORTHO)
    add_intersection(d[PROJORTHO],d[BASELINE],suffix=BASELINE)
    fig = plt.figure()
    plothist({k:d[k] for k in [PROJORTHO,BASELINE]},"domax","maximum distance between orthodromy and the trajectory\n on considered segment [m]")
    savefig(fig,f"{args.folderfigures}/domaxdist.pdf",width=6)
    fig = plt.figure()
    plothist({k:d[k] for k in [PROJORTHO,BASELINE]},"length","segment duration [s]")
    savefig(fig,f"{args.folderfigures}/lengthdist.pdf",width=6)
    fig = plt.figure()
    f = {
        PROJORTHO:f"distribution of MaxIoU(x,{BASELINE}) \nfor x in {PROJORTHO}",
        BASELINE:f"distribution of MaxIoU(x,{PROJORTHO}) \nfor x in {BASELINE}",
    }
    iou={
        PROJORTHO:BASELINE,
        BASELINE:PROJORTHO,
    }
    plothist({f[k]:d[k]for k in [PROJORTHO,BASELINE]},{f[k]:f"iou{iou[k]}" for k in [PROJORTHO,BASELINE]},"distribution of MaxIoU(x,SegmentsSet) [-]")
    savefig(fig,f"{args.folderfigures}/maxiouprojorthobaseline.pdf")
    plothist({f[k]:d[k].query("length>300") for k in [PROJORTHO,BASELINE]},{f[k]:f"iou{iou[k]}" for k in [PROJORTHO,BASELINE]},"distribution of MaxIoU(x,SegmentsSet) [-]")
    savefig(fig,f"{args.folderfigures}/maxiouprojorthobaseline300.pdf")
    add_intersection(d[PROJLOXO],d[PROJORTHO],suffix=PROJORTHO)
    # add_intersection(d[PROJORTHO],d[BASELINE],suffix=BASELINE)
    fig = plt.figure()
    df=d[PROJLOXO]
    res=df.groupby(pd.cut(df.track_start,bins=36))[f"iou{PROJORTHO}"].mean()
    n=res.index.categories.left.shape[0]
    x=np.zeros(1+n)
    x[:-1]=res.index.categories.left
    x[-1]=res.index.categories.right[-1]
    plt.stairs(res.values,x,fill=True)
    plt.xlabel("$\\text{track} [^\\circ]$")
    ystr=f"MaxIoU(x,{PROJORTHO}) for x in {PROJLOXO}\n averaged in each track bin"
    plt.ylabel(ystr)
    savefig(fig,f"{args.folderfigures}/maxiouortholoxotrack.pdf")
    fig = plt.figure()
    df=d[PROJLOXO]
    res=df.groupby(pd.cut(df.dolmax,bins=15))[f"iou{PROJORTHO}"].mean()
    n=res.index.categories.left.shape[0]
    x=np.zeros(1+n)
    x[:-1]=res.index.categories.left
    x[-1]=res.index.categories.right[-1]
    plt.stairs(res.values,x,fill=True)
    plt.xlabel("maximum distance between orthodromy and loxodromy\n on considered segment [m]")
    ystr=f"MaxIoU(x,{PROJORTHO}) for x in {PROJLOXO}\n averaged in each distance bin"
    plt.ylabel(ystr)
    savefig(fig,f"{args.folderfigures}/maxiouortholoxodolmax.pdf")
    add_intersection(d[PROJLOXONOTORTHO],d[CONFLICT],suffix=CONFLICT)
    add_intersection(d[PROJORTHONOTLOXO],d[CONFLICT],suffix=CONFLICT)
    fig = plt.figure()
    plt.hist(d[PROJORTHONOTLOXO][f"inclusion_ratio{CONFLICT}"])
    plt.xlabel(f"MaxIoL(x,{CONFLICT}) \nfor x in {PROJORTHONOTLOXO}")
    plt.ylabel(COUNT)
    savefig(fig,f"{args.folderfigures}/maxiolortho.pdf")
    fig = plt.figure()
    plt.hist(d[PROJLOXONOTORTHO][f"inclusion_ratio{CONFLICT}"])
    plt.xlabel(f"MaxIoL(x,{CONFLICT}) \nfor x in {PROJLOXONOTORTHO}")
    plt.ylabel(COUNT)
    savefig(fig,f"{args.folderfigures}/maxiolloxo.pdf")

    add_intersection(d[PROJLOXONOTORTHO],d[PROJORTHONOTLOXO],suffix=PROJORTHONOTLOXO)
    add_intersection(d[PROJORTHONOTLOXO],d[PROJLOXONOTORTHO],suffix=PROJLOXONOTORTHO)
    fig = plt.figure()
    f = {
        PROJORTHONOTLOXO:f"distribution of MaxIoU(x,{PROJLOXONOTORTHO}) \nfor x in {PROJORTHONOTLOXO}",
        PROJLOXONOTORTHO:f"distribution of MaxIoU(x,{PROJORTHONOTLOXO}) \nfor x in {PROJLOXONOTORTHO}",
    }
    iou={
        PROJLOXONOTORTHO:PROJORTHONOTLOXO,
        PROJORTHONOTLOXO:PROJLOXONOTORTHO,
    }
    toiter=[PROJORTHONOTLOXO,PROJLOXONOTORTHO]
    plothist({f[k]:d[k]for k in toiter},{f[k]:f"iou{iou[k]}" for k in toiter},"distribution of MaxIoU(x,SegmentsSet) [-]")
    savefig(fig,f"{args.folderfigures}/maxiouloxoorthoonly.pdf")
    
if __name__ == '__main__':
    main()
