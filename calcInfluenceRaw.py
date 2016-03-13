import pandas as pd
from os.path import isfile, join, dirname

DATA_PATH = join(dirname(__file__), 'dataset')
PKL_PATH = join(dirname(__file__), 'pkl')

KDD_aff = pd.read_pickle(join(PKL_PATH, 'KDD_Affl.pkl'))
KDD_paper = pd.read_pickle(join(PKL_PATH, 'KDD_Paper.pkl'))
KDD_conf = pd.read_pickle(join(PKL_PATH, 'KDD_Conf.pkl'))
KDD_conf_ins = pd.read_pickle(join(PKL_PATH, 'KDD_ConfInstance.pkl'))
KDD_PAA = pd.read_pickle(join(PKL_PATH, 'KDD_PAA.pkl'))
conf = pd.read_pickle(join(PKL_PATH, 'Conf.pkl'))

# calc author - affiliation degree
AA_matrix = {author:set() for author in KDD_PAA['Author_ID'].values}
for idx, row in KDD_PAA.iterrows():
    AA_matrix[row['Author_ID']].add(row['Affiliation_ID'])
# calc paper - affiliation influence
def calcInfluence(year):
    AFF_INF = {aff:0 for aff in KDD_aff['Affiliation_ID'].values}
    paper = KDD_paper[KDD_paper['Year']==year]
    for idx, row in paper.iterrows():
        paa = KDD_PAA[KDD_PAA['Paper_ID'] == row['Paper_ID']]
        authors = paa['Author_ID'].unique()
        for author in authors:
            affs = AA_matrix[author]
            for aff in affs:
                AFF_INF[aff] += 1.0/len(affs)*1.0/len(authors)
    return AFF_INF

# calc each affiliation's influence each year(2011-2015)
INF = {'Year':[], 'Affiliation_ID':[], 'Influence':[], 'Affiliation_Name':[]}
for year in KDD_paper['Year'].unique():
    rst = calcInfluence(year)
    for k,v in rst.iteritems():
        INF['Year'] += [year]
        INF['Affiliation_ID'] += [k]
        INF['Affiliation_Name'] += [KDD_aff[KDD_aff['Affiliation_ID']==k]['Affiliation_Name'].values[0]]
        INF['Influence'] += [v]
df = pd.DataFrame(INF)
df.to_pickle(join(PKL_PATH, 'influence.pkl'))