import numpy as np
import pandas as pd
from os.path import dirname, join


class PrepareGraph:
    def __init__(self, ConfName):
        confs = pd.read_pickle(join(join(dirname(__file__), 'pkl'), 'KDD_Conf.pkl'))
        self.confId = confs[confs['Conference_Abbrevation'] == ConfName]['Conference_ID'].iloc[0]
        data = pd.read_pickle(join(join(dirname(__file__), 'pkl'), 'KDD_ACA.pkl'))
        self.data = data[data['Conference_ID'] == self.confId]
        self.authorDict = self.encodeAuthor()
        self.N = len(self.authorDict)

    def encodeAuthor(self):
        authorDict = {}
        for idx, row in self.data.iterrows():
            authorId = row['Author_ID']
            if authorId not in authorDict.keys():
                authorDict[authorId] = len(authorDict)
        return authorDict

    def genGraph(self, year): # generating co-author graph
        graph = np.zeros((self.N, self.N))
        # pick papers at specified year
        data = self.data
        data = data[data['Year'] == year]
        # accepted label: O, X
        acceptLabel = np.zeros((self.N, 2))
        # co-author label: 1,2,3,4,5,6,7,8,9,10,10+
        coNumLabel = np.zeros((self.N, 11))
        for paperId in data['Paper_ID'].unique():
            authors = data[data['Paper_ID'] == paperId]['Author_ID'].values
            for i in range(0, len(authors)): # graph
                for j in range(i+1, len(authors)):
                    graph[self.authorDict[authors[i]], self.authorDict[authors[j]]] = 1
                # accept label
                acceptLabel[self.authorDict[authors[i]],0] = 1
                # co-author label
                c = len(authors) if len(authors) <= 10 else 11
                coNumLabel[self.authorDict[authors[i]], c-1] = 1
        return (graph, acceptLabel, coNumLabel)

class OMNIProp:
    def __init__(self, graph, accpetLabel, coNumLabel):
        pass

    def updateGraph(self, newGraph, newAcceptLabel, newCoNumLabel):
        pass

Confs = ['SIGIR', 'SIGMOD', 'SIGCOMM']
pre = PrepareGraph(Confs[0])
(G2011, acceptLabel2011, coNumLabel2011) = pre.genGraph(2011)
print(acceptLabel2011)