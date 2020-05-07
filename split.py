#!/usr/bin/env python3


# This program is used to generate our test data. It does this by 
# generating 8 random season numbers ( 20% of 39 Seasons ), and splitting
# that season's data from the training data.

import pandas as pd
import random


def main():
    rand_seasons = 8
    seasons = 39
    train_df = pd.read_csv("train-full.csv")
    nums = {}
    for i in range( rand_seasons ):
        random_season = random.randint(1,seasons)
        if ( random_season not in nums ):
            nums[random_season] = random_season
        else: 
            random_season = random.randint(1,seasons)
            if ( random_season not in nums ):
                nums[random_season] = random_season
            # hopefully it won't generate same again
            else:
                continue

        print("random season: ", random_season)
        # print(train_df[ train_df['Season'] == random_season ])
        test_df = train_df[ train_df['Season'] == random_season ]
        test_df.to_csv(f'test/test_season{random_season}.csv', index=False)
        # test_df = test_df.concat( [test_df, train_df[ train_df['Season'] == random_season ]] )
        train_df = train_df[ train_df['Season'] != random_season ]
        # train_df = train_df.drop(["Season"], axis=1)

    train_df.to_csv("train.csv", index=False)
    

    


   
if __name__ == "__main__":
    main()