import argparse
import time
import detect_classic
import detect_longest
import filter_trajs


class TimeInterval:
    def __init__(self,start,stop):
        self.start = start.timestamp()
        self.stop = stop.timestamp()
    def dt(self):
        return self.stop-self.start
    def __repr__(self):
        return f"{self.start} {self.stop} {self.dt()}"



def main():
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-trajsin')
    parser.add_argument('-detectedout')
    subparsers = parser.add_subparsers(dest="command", required=True)
    methods = {
        "longest":detect_longest.DetectLongestOrthodromyLoxodromy,
        "aligned":detect_classic.DetectOrthodromyWithBeacons,
    }
    for k in methods:
        methods[k].add_parser(subparsers.add_parser(k))
    args = parser.parse_args()
    kwargs = methods[args.command].extract_args(args)
    detector = methods[args.command](**kwargs)
    flights = filter_trajs.read_trajectories(args.trajsin)
    if flights.empty:
        flights.to_parquet(args.detectedout)
    else:
        t0 = time.time()
        res = detector.groupby_and_apply(flights)
        print(time.time()-t0)
        df=res.sort_values(by=["start"]).reset_index(drop=True)
        print(df)
        df.to_parquet(args.detectedout)

main()
