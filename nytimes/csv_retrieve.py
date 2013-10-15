import pandas as pd



df = pd.DataFrame()
for i in xrange(30):
    print "file %d" % (i+1)
    df_temp = pd.read_csv('http://stat.columbia.edu'
                            '/~rachel/datasets/nyt%d.csv' % (i+1))
    df_temp.to_csv('nytimes%d.csv' % (i+1))

# pd.concat(df, df_temp)