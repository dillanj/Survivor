#!/usr/bin/env python3
import pandas as pd

import glob


def main():
    # if (len(argv) > 1):
    #     s1 = argv[1]
    #     s2 = argv[2]
    #     s3 = argv[3]
    #     s4 = argv[4]
    #     s5 = argv[5]
    #     s6 = argv[6]
    #     s7 = argv[7]
    #     s8 = argv[8]




    path = r'/Users/dillanjohnson/Desktop/college/dixie/CS4320/final/test' # use your path
    all_files = glob.glob(path + "/*.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame = frame.sort_values( ['Season'] )
    frame.to_csv("test.csv", index=False)




    


   
if __name__ == "__main__":
        main()