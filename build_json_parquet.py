import argparse
import os
import pandas as pd
import tqdm
import read_json

def main():
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-jsonfolderin')
    parser.add_argument('-parquetout')
    args = parser.parse_args()
#    trajs = pd.read_parquet(args.trajsin)
    l = []
    d = {k:[] for k in ["start","stop","altitude","track_angle","icao24","flight_plan","flight_plan_names"]}
    for root, dirs, files in tqdm.tqdm(os.walk(args.jsonfolderin, topdown=False)):
        for name in tqdm.tqdm(files):
            #print(root,dirs,files)
            fname = os.path.join(root, name)
            json = read_json.Situation.from_json(fname)
            d["start"].append(json.deviated.start)
            d["stop"].append(json.deviated.stop)
            d["flight_plan"].append([(b.longitude,b.latitude) for b in json.deviated.beacons])
            d["flight_plan_names"].append([b.name for b in json.deviated.beacons])
            traj = json.trajectories.query("flight_id==@json.deviated.flight_id").query("timestamp==@json.deviated.start")
            assert(traj.shape[0]==1)
            for _,line in traj.iterrows():
                d["altitude"].append(line.altitude)
                d["track_angle"].append(line.track)
                d["icao24"].append(line.icao24)
            # if len(d["start"])>10:
            #     break
    df=pd.DataFrame(d).sort_values(by=["start"]).reset_index(drop=True)
    df["icao24"] = df["icao24"].astype("string")
    print(df)
    df.to_parquet(args.parquetout)


main()

