from os import listdir
from os.path import isfile, join, dirname, getsize
import pandas as pd
import re
import multiprocessing as mp

DATA_PATH = join(dirname(__file__), 'dataset')
PKL_PATH = join(dirname(__file__), 'pkl')

# Phase1 conference
CONF = ['SIGIR', 'SIGMOD', 'SIGCOMM']

# loading tranformed datasets
KDD_aff = pd.read_pickle(join(PKL_PATH, 'KDD_Affl.pkl'))
KDD_paper = pd.read_pickle(join(PKL_PATH, 'KDD_Paper.pkl'))
KDD_conf = pd.read_pickle(join(PKL_PATH, 'KDD_Conf.pkl'))
KDD_conf_ins = pd.read_pickle(join(PKL_PATH, 'KDD_ConfInstance.pkl'))
KDD_PAA = pd.read_pickle(join(PKL_PATH, 'KDD_PAA.pkl'))
conf = pd.read_pickle(join(PKL_PATH, 'Conf.pkl'))
KDD_ref = pd.read_pickle(join(PKL_PATH, 'KDD_ref.pkl'))

# filter PaperAuthorAffiliation[done]
def proc_PAA(KDD_PAA, fname, cur, end):
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
        paaLock.acquire()
        KDD_PAA['Paper_ID'] += pa
        KDD_PAA['Author_ID'] += au
        KDD_PAA['Affiliation_ID'] += af
        print('c={0}, KDD_PAA size = {1}'.format(c, len(KDD_PAA['Paper_ID'])))
        paaLock.release()

# filter PaperReferences
def proc_ref(KDD_ref, fname, cur, end):
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
        refLock.acquire()
        KDD_ref['Paper_ID'] += pa
        KDD_ref['Paper_Reference_ID'] += ref
        print('c={0}, KDD_ref size = {1}'.format(c, len(KDD_ref['Paper_ID'])))
        refLock.release()

# filter Papers
def proc_PA(KDD_PA, fname, cur, end):
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
        paLock.acquire()
        KDD_PA['Paper_ID'] += pId
        KDD_PA['Paper_Rank'] += rk
        print('c={0}, KDD_PA size = {1}'.format(c, len(KDD_PA['Paper_ID'])))
        paLock.release()

# filter PaperAuthorAffiliations[done]
# paaLock = mp.Lock()
# PAA_PATH = join(DATA_PATH, 'PaperAuthorAffiliations.txt')
# filesize = getsize(PAA_PATH)
# split_size = 1024*1024*30
# pool = mp.Pool(processes=4)
# cur = 0
# KDD_PAA = mp.Manager().dict()
# KDD_PAA['Paper_ID']=[]
# KDD_PAA['Author_ID']=[]
# KDD_PAA['Affiliation_ID']=[]

# with open(PAA_PATH) as f:
#     for chunk in xrange(filesize // split_size):
#         if cur + split_size > filesize:
#             end = filesize
#         else:
#             end = cur + split_size
#         f.seek(end)
#         f.readline()
#         end = f.tell()
#         pool.apply_async(proc_PAA, args=[KDD_PAA, PAA_PATH, cur, end])
#         cur = end
#     pool.close()
#     pool.join()

# H = ['Paper_ID', 'Author_ID', 'Affiliation_ID']
# df = pd.DataFrame({x:KDD_PAA[x] for x in H})
# df.to_pickle(join(PKL_PATH, 'KDD_PAA.pkl'))

# filter PaperReferences[done]
# refLock = mp.Lock()
# REF_PATH = join(DATA_PATH, 'PaperReferences.txt')
# filesize = getsize(REF_PATH)
# split_size = 1024*1024*30
# pool = mp.Pool(processes=4)
# cur = 0
# KDD_ref = mp.Manager().dict()
# KDD_ref['Paper_ID']=[]
# KDD_ref['Paper_Reference_ID']=[]

# with open(REF_PATH) as f:
#     for chunk in xrange(filesize // split_size):
#         if cur + split_size > filesize:
#             end = filesize
#         else:
#             end = cur + split_size
#         f.seek(end)
#         f.readline()
#         end = f.tell()
#         pool.apply_async(proc_ref, args=[KDD_ref, REF_PATH, cur, end])
#         cur = end
#     pool.close()
#     pool.join()

# H = ['Paper_ID', 'Paper_Reference_ID']
# df = pd.DataFrame({x:KDD_ref[x] for x in H})
# df.to_pickle(join(PKL_PATH, 'KDD_ref.pkl'))

# filter papers[next]
paLock = mp.Lock()
PA_PATH = join(DATA_PATH, 'Papers.txt')
filesize = getsize(PA_PATH)
split_size = 1024*1024*30
pool = mp.Pool(processes=4)
cur = 0
KDD_PA = mp.Manager().dict()
KDD_PA['Paper_ID']=[]
KDD_PA['Paper_Rank']=[]

with open(PA_PATH) as f:
    for chunk in xrange(filesize // split_size):
        if cur + split_size > filesize:
            end = filesize
        else:
            end = cur + split_size
        f.seek(end)
        f.readline()
        end = f.tell()
        pool.apply_async(proc_PA, args=[KDD_PA, PA_PATH, cur, end])
        cur = end
    pool.close()
    pool.join()

H = ['Paper_ID', 'Paper_Rank']
df = pd.DataFrame({x:KDD_PA[x] for x in H})
df.to_pickle(join(PKL_PATH, 'KDD_PA.pkl'))
