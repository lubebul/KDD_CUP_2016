import pandas as pd
from os.path import join

paa = pd.read_pickle(join('pkl', 'KDD_PAA.pkl'))
paperIds = set(paa['Paper_ID'])
for pid in paperIds:
    aas = paa[paa['Paper_ID']==pid]
    for author in set(aas['Author_ID']):
        if aas[aas['Author_ID']==author].shape[0] > 1:
               print('#Affiliation = {0} > 1'.format(aas[aas['Author_ID']==author].shape[0]))