import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage,AnnotationBbox

import process_data

def get_image(name):
    path = "castaway_pics/{}.png".format(name)
    im = plt.imread(path)
    return im

def offset_image(coord, name, ax):
    img = get_image(name)
    im = OffsetImage(img, zoom=0.72)
    im.image.axes = ax

    ab = AnnotationBbox(im, (coord, 0),  xybox=(0., -25.), frameon=False,
                        xycoords='data',  boxcoords="offset points", pad=5)

    ax.add_artist(ab)

def getResults(ldf, rdf):
    print("RDF")
    print(rdf)
    ldf.insert(0,"Probability", "0")
    for ( col_name, data ) in ldf.iteritems():
        if (col_name == "Probability"):
            for i in range( len(data) ):
                
                if len(rdf.columns) >= 2:
                    # print("into")
                    ldf[col_name][i] = rdf['1'][i] * 100
                else:
                    # print("out of", rdf.columns)
                    ldf[col_name][i] = rdf['0'] * 100
        else:   
            continue

    ldf = ldf[['playerID', 'Contestant', 'Probability']]
    return ldf

                






test_df = process_data.get_data("s40-test-updated.csv", "csv")
predictions_df = process_data.get_data("s40-predictions.csv", "csv")

results_df = getResults( test_df, predictions_df )
results_df.to_csv("s40-final-results.csv", index=False)
# print("The results df: ", results_df)

results = results_df.to_numpy()
ids = []
contestants = []
prob = []
for i in range( len( results ) ):
    ids.append( results[i][0] )
    contestants.append( results[i][1] )
    prob.append( results[i][2] )

# print( "the ids are: ", ids )
# print( "the contestants are: ", contestants )
# print( "the prob are: ", prob )

# countries = ["Norway", "Spain", "Germany", "Canada", "China"]
# valuesA = [20, 15, 30, 5, 26]


fig, ax = plt.subplots()

ax.bar(range(len(contestants)), prob, width=0.5,align="center")
ax.set_xticks(range(len(contestants)))
ax.set_xticklabels(contestants)
ax.tick_params(axis='x', which='major', pad=50)

for i, c in enumerate(ids):
    offset_image(i, c, ax)

plt.show()