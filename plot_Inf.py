import pandas as pd
from os.path import join

Inf = pd.read_pickle(join('pkl', 'Influence.pkl'))
affs = Inf['Affiliation_ID'].unique()
confs = Inf['Conference_ID'].unique()
x = {'Affiliation_ID':[]}
for conf in confs:
    for year in range(2011,2016):
        x['{0}_{1}'.format(conf,year)]=[]
for aff in affs:
    infs = Inf[Inf['Affiliation_ID']==aff]
    for idx, row in infs.iterrows():
        x['{0}_{1}'.format(row['Conference_ID'],row['Year'])] += [row['Influence']]
    x['Affiliation_ID'] += [aff]
x = pd.DataFrame(x)
x.to_pickle(join('pkl', 'Influence_By_Aff.pkl'))
x.to_csv('Influence_By_Aff.csv')