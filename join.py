#!/usr/bin/env python3

import pandas as pd


# this program is used to comibine all the data files into one csv matched on playerID



def main():
    # idols_df = pd.read_csv("all-idols.csv")
    # mpf_df = pd.read_csv("all-mpf.csv")
    # survivor_df = pd.read_csv("all-data.csv")

    # # all_df = pd.merge( survivor_df, idols_df, on=["playerID", "Season"], how="left")
    # all_df = pd.merge( survivor_df, mpf_df, on=["playerID", "Season"], how="left")
    # all_df.to_csv("all-combined.csv", index=False)


    # # cut_df = all_df[['playerID', 'Season', 'Contestant_x', 'Winner_x', 'SurvSc', 'SurvAv', 'Days', 'Finish', 'Time',  'ChW', 'ChA', 'ChW%', 'SO', 'MPF','VAP', 'TotV', 'TCA', 'TC%', 'wTCR', 'JVF', 'TotJ', 'JV%_x', 'Idols found', 'Idols played', 'Votes voided', 'Boot avoided', 'Day found', 'Day played']]
    # cut_df = all_df[['playerID', 'Season', 'Contestant_x', 'Winner_x', 'SurvSc', 'SurvAv', 'Days', 'Finish', 'Time',  'ChW', 'ChA', 'ChW%', 'SO', 'MPF','VAP', 'TotV', 'TCA', 'TC%', 'wTCR', 'JVF', 'TotJ', 'JV%_x']]
    # cut_df = cut_df.rename(mapper={"Contestant_x": "Contestant", "Winner_x": "Winner", "JV%_x": "JV%" }, axis=1)
    # cut_df.to_csv("train-full.csv", index=False)

    mpf_s40 = pd.read_csv("s40-mpf.csv")
    survivor_s40 = pd.read_csv("s40-data.csv")

    s40_df = pd.merge( survivor_s40, mpf_s40, on=["playerID"], how="left")
    s40_df = s40_df.rename(mapper={"Contestant_x": "Contestant", "ChA_x": "ChA", "ChW.1": "ChW%" }, axis=1)
    s40_df = s40_df.drop( columns=["Contestant_y", "ChA_y"])
    s40_df.to_csv("s40-test.csv", index=False)



if __name__ == "__main__":
    main()