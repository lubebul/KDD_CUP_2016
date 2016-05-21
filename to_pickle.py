from os import listdir
from os.path import isfile, join, dirname, getsize
import pandas as pd
import re
import multiprocessing as mp

DATA_PATH = join(dirname(__file__), 'dataset')
PKL_PATH = join(dirname(__file__), 'pkl')
# Phase1 conference
CONF = ['SIGIR', 'SIGMOD', 'SIGCOMM']
# Phase2 conference
CONF = ['KDD', 'ICML']
# Phase3 conference
CONF = ['FSE', 'MOBICOM', 'MM']

def select_from_df(df, column, targets):
    tmp = None
    for x in targets:
        print(x)
        tmp = df[df[column]==x] if tmp is None else tmp.append(df[df[column]==x])
    return tmp

def get_conf():
    conf = pd.read_pickle(join(PKL_PATH, 'Conf.pkl'))
    df = select_from_df(conf, 'Conference_Abbrevation', CONF)
    df.to_pickle(join(PKL_PATH, 'KDD_Conf.pkl'))
    confIns = pd.read_table(join(DATA_PATH, 'ConferenceInstances.txt'), names=['Conference_ID', 'Conference_Instance_ID', 'Conference_Abbrevation', 'Conference_Full_Name', 'Location', 'URL', 'start_date', 'end_date', 'registration_date', 'submission_deadline', 'notification_due', 'final_version_due'])
    confId = list(set(df['Conference_ID'].values))
    df = select_from_df(confIns, 'Conference_ID', confId)
    df.to_pickle(join(PKL_PATH, 'KDD_ConfInstance.pkl'))


# select conference
# get_conf()

# loading tranformed datasets
conf = pd.read_pickle(join(PKL_PATH, 'Conf.pkl'))
KDD_aff = pd.read_pickle(join(PKL_PATH, 'KDD_Affl.pkl'))
KDD_paper = pd.read_pickle(join(PKL_PATH, 'KDD_Paper.pkl'))
KDD_conf = pd.read_pickle(join(PKL_PATH, 'KDD_Conf.pkl'))
KDD_conf_ins = pd.read_pickle(join(PKL_PATH, 'KDD_ConfInstance.pkl'))
ACA = pd.read_pickle(join(PKL_PATH, 'KDD_ACA.pkl'))

DF = None

def filter(txt_path, columns, proc_fun):
    REF_PATH = join(DATA_PATH, txt_path)
    filesize = getsize(REF_PATH)
    split_size = 1024*1024*30
    pool = mp.Pool(processes=4)
    cur = 0
    DF = mp.Manager().dict()
    for x in columns:
        DF[x] = []
    with open(REF_PATH) as f:
        for chunk in xrange(filesize // split_size):
            end = filesize if cur + split_size > filesize else cur + split_size
            f.seek(end)
            f.readline()
            end = f.tell()
            pool.apply_async(proc_fun, args=[DF, REF_PATH, cur, end])
            cur = end
        pool.close()
        pool.join()
    df = pd.DataFrame({x:DF[x] for x in columns})
    return df

# filter PaperAuthorAffiliation
def proc_PAA(DF, fname, cur, end):
    # use set to speedup lookup operation
    KDD_Papers = set(KDD_paper['Paper_ID'].values)
    AFFs = set(KDD_aff['Affiliation_ID'].values)
    with open(fname, 'r') as f:
        f.seek(cur)
        data = f.read(end-cur)
    paperIds = re.findall('(\w*)\t\w*\t\w*\t.*\t.*\t\d*', data)
    authorIds = re.findall('\w*\t(\w*)\t\w*\t.*\t.*\t\d*', data)
    affIds = re.findall('\w*\t\w*\t(\w*)\t.*\t.*\t\d*', data)
    c = 0
    pa,au,af = [],[],[]
    for i in range(len(paperIds)):
        if paperIds[i] in KDD_Papers and affIds[i] in AFFs:
            pa += [paperIds[i]]
            au += [authorIds[i]]
            af += [affIds[i]]
            c += 1
    if c > 0:
        with LOCK:
            DF['Paper_ID'] += pa
            DF['Author_ID'] += au
            DF['Affiliation_ID'] += af
            print('c={0}, KDD_PAA size = {1}'.format(c, len(DF['Paper_ID'])))

# filter PaperReferences
def proc_ref(DF, fname, cur, end):
    # use set to speedup lookup: O(1)
    KDD_Papers = set(KDD_paper['Paper_ID'].values)
    with open(fname, 'r') as f:
        f.seek(cur)
        data = f.read(end-cur)
    paperIds = re.findall('(\w*)\t\w*', data)
    refIds = re.findall('\w*\t(\w*)', data)
    c = 0
    pa,ref = [],[]
    for i in range(len(paperIds)):
        if paperIds[i] in KDD_Papers or refIds[i] in KDD_Papers:
            pa += [paperIds[i]]
            ref += [refIds[i]]
            c += 1
    if c > 0:
        with LOCK:
            DF['Paper_ID'] += pa
            DF['Paper_Reference_ID'] += ref
            print('c={0}, KDD_ref size = {1}'.format(c, len(DF['Paper_ID'])))

# filter author
def proc_Author(DF, fname, cur, end):
    authorSet = set(ACA['Author_ID'].values)
    with open(fname, 'r') as f:
        f.seek(cur)
        data = f.read(end-cur)
    authorIds = re.findall('(.*)\t.*\n', data)
    authorNames = re.findall('.*\t(.*)\n', data)
    c = 0
    aid, aName = [],[]
    for i in range(len(authorIds)):
        if authorIds[i] in authorSet:
            aid += [authorIds[i]]
            aName += [authorNames[i]]
            c += 1
    print('Proc {0} authorIds, found {1} in selected author'.format(len(authorIds), c))
    if c > 0:
        with LOCK:
            DF['Author_ID'] += aid
            DF['Author_Name'] += aName
            print('c={0}, Author size = {1}'.format(c, len(DF['Author_ID'])))

# filter Papers
def proc_PA(DF, fname, cur, end):
    # use set to speedup lookup operation
    Ref1 = set(KDD_ref['Paper_ID'].values)
    Ref2 = set(KDD_ref['Paper_Reference_ID'].values)
    with open(fname, 'r') as f:
        f.seek(cur)
        data = f.read(end-cur)
    paperIds = re.findall('(.*)\t.*\t.*\t.*\t.*\t.*\t.*\t.*\t.*\t.*\t.*\n', data)
    rank = re.findall('.*\t.*\t.*\t.*\t.*\t.*\t.*\t.*\t.*\t.*\t(.*)\n', data)
    c = 0
    pId,rk = [],[]
    for i in range(len(paperIds)):
        if paperIds[i] in Ref1 or paperIds[i] in Ref2:
            pId += [paperIds[i]]
            rk += [rank[i]]
            c += 1
    if c > 0:
        with LOCK:
            DF['Paper_ID'] += pId
            DF['Paper_Rank'] += rk
            print('c={0}, KDD_PA size = {1}'.format(c, len(DF['Paper_ID'])))

# filter Papers
def proc_Keyword(DF, fname, cur, end):
    KDD_PA = pd.read_pickle(join(PKL_PATH, 'KDD_PA.pkl'))
    # use set to speedup lookup operation
    paperId = set(KDD_PA['Paper_ID'].values)
    with open(fname, 'r') as f:
        f.seek(cur)
        data = f.read(end-cur)
    paperIds = re.findall('(.*)\t.*\t.*\n', data)
    keywords = re.findall('.*\t(.*)\t.*\n', data)
    fieldIds = re.findall('.*\t.*\t(.*)\n', data)
    c = 0
    pId, kw, fid = [], [], []
    for i in range(len(paperIds)):
        if paperIds[i] in paperId:
            pId += [paperIds[i]]
            kw += [keywords[i]]
            fid += [fieldIds[i]]
            c += 1
    if c > 0:
        with LOCK:
            DF['Paper_ID'] += pId
            DF['Keyword_Name'] += kw
            DF['Field_ID'] += fid
            print('c={0}, PaperKeyword size = {1}'.format(c, len(DF['Paper_ID'])))

LOCK = mp.Lock()
# filter PaperAuthorAffiliations
KDD_PAA = filter('PaperAuthorAffiliations.txt', ['Paper_ID', 'Author_ID', 'Affiliation_ID'], proc_PAA)
KDD_PAA.to_pickle(join(PKL_PATH, 'KDD_PAA.pkl'))

# filter PaperReferences
KDD_ref = filter('PaperReferences.txt', ['Paper_ID', 'Paper_Reference_ID'], proc_ref)
KDD_ref.to_pickle(join(PKL_PATH, 'KDD_ref.pkl'))

# filter papers
KDD_PA = filter('Papers.txt', ['Paper_ID', 'Paper_Rank'], proc_PA)
KDD_PA.to_pickle(join(PKL_PATH, 'KDD_PA.pkl'))

# filter authors
KDD_A = filter('Authors.txt', ['Author_ID', 'Author_Name'], proc_Author)
KDD_A.to_pickle(join(PKL_PATH, 'Authors.pkl'))

# keyword vector
# Paper_Keyword = filter('PaperKeywords.txt', ['Paper_ID', 'Keyword_Name', 'Field_ID'], proc_Keyword)
# Paper_Keyword.to_pickle(join(PKL_PATH, 'PaperKeywords.pkl'))