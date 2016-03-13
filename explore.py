import pandas as pd
import re
from os.path import isfile, join, dirname

DATA_PATH = join(dirname(__file__), 'dataset')
PKL_PATH = join(dirname(__file__), 'pkl')

KDD_aff = pd.read_pickle(join(PKL_PATH, 'KDD_Affl.pkl'))
KDD_paper = pd.read_pickle(join(PKL_PATH, 'KDD_Paper.pkl'))
KDD_conf = pd.read_pickle(join(PKL_PATH, 'KDD_Conf.pkl'))
KDD_conf_ins = pd.read_pickle(join(PKL_PATH, 'KDD_ConfInstance_Raw.pkl'))
KDD_PAA = pd.read_pickle(join(PKL_PATH, 'KDD_PAA.pkl'))
conf = pd.read_pickle(join(PKL_PATH, 'Conf.pkl'))
inf = pd.read_pickle(join(PKL_PATH, 'influence.pkl'))

rankAff = inf[inf['Year']==2015].sort(['Influence'], ascending=[0])
for i in range(10):
    print(inf[inf['Affiliation_ID']==rankAff.iloc[i]['Affiliation_ID']].sort(['Year'], ascending=[1]))