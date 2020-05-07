#!/usr/bin/env python3


############# PURPOSE ################


# THIS FILE WAS USED TO FORMAT EACH OF THE THREE CSV DATA FILES THAT I AM USING 
# WITH A PLAYER ID. THIS ALLOWED FOR RETURN PLAYERS TO BE VIEWED AS SUCH, NOT A COMPLETE
# NEW PLAYER. THIS ALSO TOOK CARE OF ANY INCONSISTENCIES BETWEEN NAMES AMONGST THE SEPERATE FILES


############# PURPOSE ################

import pandas as pd

CONTESTANTS = {}



def generate_player_id(df):
    player_id = 0000
    df.insert(0,"playerID", "NaN")
    # df.insert(3, "Winner", 0)
    for ( col_name, data ) in df.iteritems():
        if (col_name == "Contestant"):
            for i in range( len(data)):
                key = data[i].lower()
                key = key.replace('.', '')
                key = key.replace('-', '')
                key = key.replace('1', '')
                key = key.replace('2', '')
                key = key.replace('3', '')
                key = key.replace('4', '')
                key = key.replace('5', '')
                key = key.replace('"', '')
                key = key.replace('*', '')
                keys = key.split()
                key = keys[0] + "-" + keys[len(keys) - 1]
                if key not in CONTESTANTS:
                    player_id += 1
                    CONTESTANTS[key] = player_id
                df['playerID'][i] = CONTESTANTS[key]  
        # elif (col_name == "Finish"):
        #     for i in range( len(data) ):
        #         if ( data[i] == 1 ):
        #             df['Winner'][i] = 1
                    # print("we have a winner")    
        else:
            continue
    
    
    return 


def prepSeasonCol( df ):
    for ( col_name, data ) in df.iteritems():
        if (col_name == "Season"):
            for i in range( len(data) ):
                df[col_name][i] = data[i].strip('S')

    return df


def prepMPFCol( df ):
    for ( col_name, data ) in df.iteritems():
        if (col_name == "MPF"):
            for i in range( len(data) ):
                df[col_name][i] = data[i].strip('%')
                df[col_name][i] = float(df[col_name][i]) * .01

    return df
    




def main():
    idols_df = pd.read_csv("original_data/idols.csv")
    mpf_df = pd.read_csv("original_data/mpf.csv")
    survivor_df = pd.read_csv("original_data/seasons.csv")

    generate_player_id( survivor_df)
    survivor_df.to_csv("all-data.csv", index=False)

    idols_df = prepSeasonCol(idols_df)
    generate_player_id(idols_df)
    idols_df.to_csv("all-idols.csv", index=False)

    mpf_df = prepSeasonCol(mpf_df)
    mpf_df = prepMPFCol(mpf_df)
    generate_player_id(mpf_df)
    mpf_df.to_csv("all-mpf.csv", index=False)

    survivor_s40 = pd.read_csv("s40/s40_data.csv")
    mpf_s40 = pd.read_csv("s40/s40_mpf.csv")
    generate_player_id( survivor_s40 )
    generate_player_id( mpf_s40 )
    mpf_s40 = prepMPFCol( mpf_s40 )
    mpf_s40.to_csv("s40-mpf.csv", index=False)
    survivor_s40.to_csv("s40-data.csv", index=False)






    # print(CONTESTANTS)
    # print("Survivor DF", survivor_df)
    # print("The Contestant Dict: ", CONTESTANTS)





if __name__ == "__main__":
    main()