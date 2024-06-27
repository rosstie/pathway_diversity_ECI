import pandas as pd

# Load the data and take 100 randome lines of it
hs92 = pd.read_csv('../data/hs92_proximities.csv')


hs92_100 = hs92.sample(100)
#save to a new file
hs92_100.to_csv('../data/hs92_proximities_100.csv', index=False)

hs92_1000 = hs92.sample(1000)
#save to a new file
hs92_1000.to_csv('../data/hs92_proximities_1000.csv', index=False)

hs92_10000 = hs92.sample(10000)
#save to a new file
hs92_10000.to_csv('../data/hs92_proximities_10000.csv', index=False)
