import pandas as pd
import optparse
import re


def parse_args():
    parser = optparse.OptionParser('usage%prog [-i filename to parse]')
    parser.add_option('-i', dest='filename', help='filename to parse')
    (options, args) = parser.parse_args()
    return options

opts = parse_args()
if opts.filename is not None:
    with open(opts.filename, 'r') as f:
        data = f.read()
papers = re.findall('.*\n(.*)\n\n', data)
last = re.findall('\n.*\n(.*)', data)
papers.append(last[-1])
Affs = pd.read_pickle('../../pkl/KDD_Affl.pkl')
score = {}
name={}
for paper in papers:
    authors = re.findall('(\w* \w*) \(\w*[ \w*]*\)', paper)
    affs = re.findall('\w* \w* \((\w*[ \w*]*)\)', paper)
    n = len(authors)
    for aff in affs:
        affId = Affs[Affs['Affiliation_Name']==aff.lower()]
        if len(affId) > 0:
            aid = affId['Affiliation_ID'].iloc[0]
            score[aid] = score[aid] + 1/float(n) if aid in score.keys() else 1/float(n)
            name[aid] = affId['Affiliation_Name'].iloc[0]
for idx, row in Affs.iterrows():
    aid = row['Affiliation_ID']
    if aid not in score:
        score[aid] = 0
        name[aid] = row['Affiliation_Name']
ids, names, scores = score.keys(), [], []
for s in ids:
    names += [name[s]]
    scores += [score[s]]
x = pd.DataFrame({'Affiliation_ID':ids, 'Affiliation_Name':names, 'Score':scores})
x = x.sort(['Score'],ascending=[0])
x.to_csv('SIGIR.csv')

pred = pd.read_table('results.tsv', names=['Conference_ID','Affiliation_ID','Score'])
confs = pd.read_pickle('../../pkl/Conf.pkl')
confId = confs[confs['Conference_Abbrevation']=='SIGIR']['Conference_ID'].iloc[0]
pred = pred[pred['Conference_ID']==confId]
pred = pred.sort(['Score'],ascending=[0])
aff_name = []
for idx, row in pred.iterrows():
    aff_name.append(name[row['Affiliation_ID']])
pred['Affiliation_Name'] = aff_name
pred = pred.drop('Conference_ID',1)
pred.to_csv('predition.csv')










