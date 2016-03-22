import pandas as pd
import numpy as np
from os.path import dirname, join
from sklearn.ensemble import RandomForestRegressor

PKL_PATH = join(dirname(__file__), 'pkl')

KDD_affs = pd.read_pickle(join(PKL_PATH, 'KDD_Affl.pkl'))
KDD_conf = pd.read_pickle(join(PKL_PATH, 'KDD_Conf.pkl'))
KDD_ACA = pd.read_pickle(join(PKL_PATH, 'KDD_ACA.pkl'))
Influence = pd.read_pickle(join(PKL_PATH, 'Influence.pkl'))


def encoding(df, dic, rev, keyname, c):
    IDs = []
    for idx, row in df.iterrows():
        aid = row[keyname]
        if aid in dic.keys():
            IDs.append(dic[aid])
        else:
            dic[aid] = c
            rev[c] = aid
            IDs.append(c)
            c += 1
    return (dic, rev, IDs, c)
def mapBack(lst, dic):
    rst = []
    for i in lst:
        rst.append(dic[i])
    return rst

for cid, crow in KDD_conf.iterrows():
    print(crow['Conference_Abbrevation'])
    cInf = Influence[Influence['Conference_ID']==crow['Conference_ID']]
    (ID_Dict, ID_Rev_Dic, Aff_TMP_ID, c) = encoding(cInf, {}, {}, 'Affiliation_ID', 0)
    Basic_Input = pd.DataFrame({'ID':Aff_TMP_ID, 'Year':cInf['Year']})

    authorReg = RandomForestRegressor(n_estimators=10).fit(Basic_Input, cInf['Author_Num'])
    coauthorReg = RandomForestRegressor(n_estimators=10).fit(Basic_Input, cInf['Co_Author_Avg'])

    cInf['ID'] = Aff_TMP_ID
    infReg = RandomForestRegressor(n_estimators=10).fit(cInf[['ID','Year','Author_Num','Co_Author_Avg']], cInf['Influence'])

    (ID_Dict, ID_Rev_Dic, KDD_Aff_ID, c) = encoding(KDD_affs, ID_Dict, ID_Rev_Dic, 'Affiliation_ID', c)
    Pred_Basic_Input = pd.DataFrame({'ID':KDD_Aff_ID, 'Year':np.repeat([2016], len(KDD_Aff_ID))})

    author = authorReg.predict(Pred_Basic_Input)
    coauthor = coauthorReg.predict(Pred_Basic_Input)

    Pred_Input = pd.DataFrame({'ID':KDD_Aff_ID,'Year':Pred_Basic_Input['Year'],'Author_Num':author,'Co_Author_Avg':coauthor})
    inf = infReg.predict(Pred_Input)
    result = pd.DataFrame({'Affiliation_ID':mapBack(KDD_Aff_ID,ID_Rev_Dic),'Influence':inf})
    inf2015 = []
    cinf2015 = cInf[cInf['Year']==2015]
    for idx, row in result.iterrows():
        x = cinf2015[cinf2015['Affiliation_ID']==row['Affiliation_ID']]
        inf2015.append(x.iloc[0]['Influence'])
    result['Influence_2015'] = inf2015
    result.to_csv('{0}_Influence.csv'.format(crow['Conference_Abbrevation']))