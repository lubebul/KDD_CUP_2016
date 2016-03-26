import pandas as pd
import numpy as np
from os.path import join
import optparse


def parse_arg():
    parser = optparse.OptionParser('usage%prog [-i inputDir] [-o outputDir]')
    parser.add_option('-i', dest='inputDir', default='')
    parser.add_option('-o', dest='outputDir', default='')
    (options, args) = parser.parse_args()
    return options

confs = pd.read_pickle(join('pkl', 'KDD_Conf.pkl'))
aca = pd.read_pickle(join('pkl', 'KDD_ACA.pkl'))
aff = pd.read_pickle(join('pkl', 'KDD_Affl.pkl'))
affs = []
prob = []
conf = []
opts = parse_arg()
INPUT_DIR, OUTPUT_DIR = opts.inputDir, opts.outputDir

for confId in confs['Conference_ID'].values:
    tprob = []
    omni_rst = pd.read_pickle(join(INPUT_DIR, 'OMNI_result_{0}.pkl'.format(confId)))
    for affId in aff['Affiliation_ID'].values:
        affScore = 0.0
        authors = aca[aca['Affiliation_ID'] == affId]['Author_ID']
        for authorId in authors.values:
            upC, upA, downC, downA = 0.0, 0.0, 0.0, 0.0
            author = omni_rst[omni_rst['Author_ID'] == authorId]
            if author.shape[0] > 0:
                author = author.iloc[0]
            else:
                continue
            for i in range(11):
                upA += author['acceptNum_self_{0}'.format(i)]*i
                upC += author['coAuthorNum_self_{0}'.format(i+1)]*(1.0/(i+1))
                downA += author['acceptNum_self_{0}'.format(i)]
                downC += author['coAuthorNum_self_{0}'.format(i+1)]
            affScore += (upA/downA)*(upC/downC)
        affs += [affId]
        tprob += [affScore]
        conf += [confId]
    m = max(tprob)
    prob += [tprob[i]/m for i in range(len(tprob))]


df = pd.DataFrame({'1_Conference_ID':conf, '2_Affiliation_ID':affs, '3_Score':prob})
df.to_csv(join(OUTPUT_DIR, 'results.tsv'), header=False, index=False, sep='\t')