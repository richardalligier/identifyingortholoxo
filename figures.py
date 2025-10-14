import matplotlib.pyplot as plt
from collections import namedtuple
import pandas as pd

def read_config():
    ''' read config to get folders '''
    with open("CONFIG","r") as f:
        l = [[x.strip() for x in line.split("=")] for line in f if len(line.strip())>0 and line.strip()[0]!="#"]
        d = {k:v for k,v in l}
    Config = namedtuple("config",list(d))
    return Config(**d)

def main():
    config = read_config()
    print(config)
    traj = pd.read_parquet(f"{config.FOLDER}/trajs/2022-07-14.parquet")
    print(traj)


if __name__=='__main__':
    main()
