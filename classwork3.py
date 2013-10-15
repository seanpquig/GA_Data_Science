# Import libraries
import numpy
import pandas as pd

# populate DataFrame object with csv files
df = pd.DataFrame()
for i in xrange(30):
    print "importing file %d..." % (i+1)
    df_temp = pd.read_csv('http://stat.columbia.edu'
                          '/~rachel/datasets/nyt%d.csv' % (i+1))
    # df_temp = pd.read_csv('/Users/seanpquig/Documents/'
    #         'GA_Data_Science/nytimes/nytimes%d.csv' % (i+1))
    df = df.append(df_temp, ignore_index=True)


# new DataFrame to get data aggregated by Age, Gender, Signed_In
dfg = df[['Age', 'Gender', 'Signed_In', 'Clicks', 'Impressions']]\
    .groupby(['Age', 'Gender', 'Signed_In']).agg([numpy.sum])
# Add a column for CTR to dfg
dfg['CTR'] = dfg['Clicks'].astype(float) / dfg['Impressions']

dfg.to_csv('nytimes_aggregation.csv')
print "CSV output complete."
