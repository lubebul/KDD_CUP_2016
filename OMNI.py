import numpy as np
import pandas as pd
from os.path import join
import snap

class PrepareGraph:
    def __init__(self, ConfName):
        confs = pd.read_pickle(join('pkl', 'KDD_Conf.pkl'))
        self.confId = confs[confs['Conference_Abbrevation'] == ConfName]['Conference_ID'].iloc[0]
        data = pd.read_pickle(join('pkl', 'KDD_ACA.pkl'))
        self.data = data[data['Conference_ID'] == self.confId]
        (self.authorDict, self.authorNameDict, self.authorIdDict) = self.encodeAuthor()
        self.N = len(self.authorDict)

    def encodeAuthor(self):
        authorDict = {}
        authorNameDict = {}
        authorIdDict = {}
        author = pd.read_pickle(join('pkl', 'Authors.pkl'))
        authors = set(author['Author_ID'].values)
        for idx, row in self.data.iterrows():
            authorId = row['Author_ID']
            if authorId not in authorDict.keys():
                aid = len(authorDict)
                authorDict[authorId] = aid
                authorNameDict[aid] = author[author['Author_ID'] == authorId]['Author_Name'].iloc[0] if authorId in authors else 'Unknown'
                authorIdDict[aid] = authorId
        return (authorDict, authorNameDict, authorIdDict)

    def genGraph(self): # generating co-author graph
        graph = np.zeros((self.N, self.N))
        # pick papers at specified year
        data = self.data
        for paperId in data['Paper_ID'].unique():
            authors = data[data['Paper_ID'] == paperId]['Author_ID'].values
            for i in range(0, len(authors)): # graph
                for j in range(i+1, len(authors)):
                    graph[self.authorDict[authors[i]], self.authorDict[authors[j]]] += 1
        return graph

    def genLabel(self, year=None):
        data = self.data
        if year is not None:
            data = data[data['Year'] == year]
        # accepted label: 0,1,2,...,9,10+
        acceptNum = np.zeros((self.N, 1))
        # co-author label[mixed]: 1,2,3,4,5,6,7,8,9,10,10+
        coNumLabel = np.zeros((self.N, 11))
        for paperId in data['Paper_ID'].unique():
            authors = data[data['Paper_ID'] == paperId]['Author_ID'].values
            for i in range(0, len(authors)):
                # accept label
                acceptNum[self.authorDict[authors[i]], 0] += 1
                # co-author label
                c = len(authors) if len(authors) <= 10 else 11
                coNumLabel[self.authorDict[authors[i]], c-1] += 1
        acceptLabel = np.zeros((self.N, 11))
        for i in range(self.N):
            c = int(acceptNum[i]) if acceptNum[i] < 10 else 10
            acceptLabel[i,c] = 1
        return (acceptLabel, coNumLabel)

    def plotGraph(self, graph, outfile):
        G = snap.TUNGraph.New()
        for i in range(self.N):
            G.AddNode(i)
        # add name to nodes
        S = snap.TIntStrH()
        for i in range(self.N):
            for j in range(i+1, self.N):
                for x in range(0,int(graph[i,j])):
                    G.AddEdge(i,j)
            S.AddDat(i, self.authorNameDict[i])
        snap.DrawGViz(G, snap.gvlDot, outfile, "Graph", S)
        
class OMNIProp:
    def __init__(self, confName, lamda, eta):
        self.pre = PrepareGraph(confName)
        self.N = self.pre.N
        self.graph = self.pre.genGraph()
        self.lamda, self.eta = lamda, eta
        self.genPrior()

    def run(self, st_year, ed_year):
        cSum = 1
        for year in range(st_year,ed_year+1):
            self.lamda = self.lamda * (year-st_year+1) / cSum
            print('[{0} Year]'.format(year))
            if year == st_year:
                self.initParam(year)
            else:
                self.updateParam(year)
            self.prop()
            cSum += year-st_year+2
        return (self.AS, self.AT, self.CS, self.CT)

    def genLabeledDict(self, ALabel):
        return {i: True if ALabel[i,0] == 0 else False for i in range(self.N)}

    def prop(self):
        # AcceptNumLabel
        diff = 1
        while diff > 1e-6:
            AT = self.iterateT(self.AS, self.BA)
            AS, ASU = self.iterateS(self.AS, self.AT, self.BA)
            diff = np.linalg.norm(AT-self.AT) + np.linalg.norm(ASU-self.ASU)
            self.AT, self.AS, self.ASU = AT, AS, ASU
            # print('acceptLabel iter diff={0}'.format(diff))
        # CoAuthorNumLabel
        diff = 1
        while diff > 1e-6:
            CT = self.iterateT(self.CS, self.BC)
            CS, CSU = self.iterateS(self.CS, self.CT, self.BC)
            diff = np.linalg.norm(CT-self.CT) + np.linalg.norm(CSU-self.CSU)
            self.CT, self.CS, self.CSU = CT, CS, CSU
            # print('coAuthorLabel iter diff={0}'.format(diff))

    def genPrior(self): # avg (per year)
        s, t = np.array([0.0 for x in range(11)]), np.array([0.0 for x in range(11)])
        for year in range(2011,2016):
            (ALabel, CoLabel) = self.pre.genLabel(year=year)
            x = np.array([sum(ALabel[:,i]) for i in range(ALabel.shape[1])])
            s += x/sum(x)
            x = np.array([sum(CoLabel[:,i]) for i in range(CoLabel.shape[1])])
            t += x/sum(x)
        s /= 5
        t /= 5
        self.BA = s # prior for AcceptNumLabel
        self.BC = t # prior for CoNumLabel

    def getUnlabeledS(self):
        ASU, CSU = [], []
        for i in range(len(self.UDict)):
            ASU.append(self.AS[self.UDict[i],:])
            CSU.append(self.CS[self.UDict[i],:])
        return ASU, CSU

    def updateUnlabeledS(self, S, SU):
        SS = S.copy()
        for i in range(len(self.UDict)):
            SS[self.UDict[i],:] = SU[i,:]
        return SS

    def initParam(self, year):
        self.AS, self.AT, self.CS, self.CT, self.UDict = self.getParam(self.BA, self.BC, year)
        self.ASU, self.CSU = self.getUnlabeledS()
        self.graphU = self.getUnlabeledGraph()

    def updateParam(self, year):
        AS, AT, CS, CT, self.UDict = self.getParam(self.BA, self.BC, year)
        self.AS = self.AS*(1-self.eta) + AS*self.eta
        self.AT = self.AT*(1-self.eta) + AT*self.eta
        self.CS = self.CS*(1-self.eta) + CS*self.eta
        self.CT = self.CT*(1-self.eta) + CT*self.eta
        self.ASU, self.CSU = self.getUnlabeledS()
        self.graphU = self.getUnlabeledGraph()

    def getUnlabeledGraph(self):
        UG = []
        for i in range(len(self.UDict)):
            UG.append(self.graph[self.UDict[i],:])
        return np.array(UG)

    def getParam(self, BA, BC, year):
        AT, CT = np.array([BA for x in range(self.N)]), np.array([BC for x in range(self.N)])
        self.ALabel, self.CLabel = self.pre.genLabel(year=year)
        self.Dict = self.genLabeledDict(self.ALabel)
        UDict = {}
        AS, CS = [], []
        for i in range(self.N):
            if not self.Dict[i]:
                UDict[len(UDict)] = i
                AS.append(BA)
                CS.append(BC)
            else:
                AS.append(self.ALabel[i,:])
                CS.append(self.CLabel[i,:])
        AS = np.array(AS)
        CS = np.array(CS)
        return (AS, AT, CS, CT, UDict)

    def iterateS(self, S, T, B): # only update unlabeled points
        # S_ik = (sum_j(A_ij*T_jk) + lamda*B_k) / (sum_j(A_ij) + lamda)
        DU = np.diag(np.array(1.0/(self.lamda+self.graphU.sum(1))))
        X = np.dot(self.graphU, T) + self.lamda * (np.ones((len(self.UDict), 1)) * np.asmatrix(B))
        SU = np.dot(DU, X)
        return (self.updateUnlabeledS(S, SU), SU)
        
    def iterateT(self, S, B):
        # T_jk = (sum_i(A_ij*S_ik) + lamda*B_k) / (sum_i(A_ij) + lamda)
        F = np.diag(np.array(1.0/(self.lamda+self.graph.sum(0))))
        X = np.dot(np.transpose(self.graph), S) + self.lamda * (np.ones((self.N, 1)) * np.asmatrix(B))
        return np.dot(F,X)

Confs = ['ICML', 'KDD']
Eta = [0.4,0.25,0.5]
for x in range(len(Confs)):
    print(Confs[x])
    omni = OMNIProp(Confs[x], lamda=1.0, eta=Eta[x])
    (AS,AT,CS,CT) = omni.run(2011,2015)

    data = {}
    for i in range(11):

        data['acceptNum_self_{0}'.format(i)] = AS[:,i].tolist()
        data['acceptNum_t_{0}'.format(i)] = AT[:,i].tolist()
        data['coAuthorNum_self_{0}'.format(i+1)] = CS[:,i].tolist()
        data['coAuthorNum_t_{0}'.format(i+1)] = CT[:,i].tolist()
    data['Author_ID'] = pd.Series([omni.pre.authorIdDict[x] for x in range(omni.N)])
    data['Author_Name'] = pd.Series([omni.pre.authorNameDict[x] for x in range(omni.N)])
    df = pd.DataFrame(data)
    df.to_pickle('OMNI_result_{0}.pkl'.format(omni.pre.confId))

