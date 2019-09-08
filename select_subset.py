import pandas as pd

CLASSES = 50
SAMPLES_PER_CLASS = 1500

# Load the train.csv file
df = pd.read_csv('datasets/train.csv', delimiter=',')

# drop invalid entries
df = df.mask(df.eq('None')).dropna()

# drop entries containing panoramio URLs because it doesn't work
df = df[~df.url.str.contains("panoramio")]

# find the most popular landmarks and keep the top {CLASSES}
mostPopular = df.groupby('landmark_id')['landmark_id'].count().reset_index(name='count').sort_values(['count'], ascending=False).head(CLASSES)

# For each of the landmarks, keep {SAMPLES_PER_CLASS}
subsets = []
for i in mostPopular['landmark_id']:
    rs = df[df['landmark_id'] == i]
    rs = rs.sample(SAMPLES_PER_CLASS).reset_index(drop=True)
    subsets.append(rs)

# Create one big dataframe and save it to disk
pd.concat(subsets, axis=0).to_csv('datasets/subset.csv', sep=',', header=True)
