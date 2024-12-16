


import os, sys, pandas, openpyxl, numpy, json, random, glob, scipy, tempfile, itertools
import sklearn
from sklearn import preprocessing

from openbabel import pybel
from openbabel import openbabel

from rdkit import rdBase, Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem, Draw, PandasTools, Descriptors, FilterCatalog, rdCoordGen, rdMolTransforms
from rdkit.Chem.Draw import IPythonConsole
import cctk
from pyscf import gto # , scf, dft
import progressbar
import concurrent #  from concurrent import futures

import matplotlib.pyplot as plt



def OpenpyxlReadWorkbook(
    ofp, 
    *, # 以下はデフォルトパラメータ
    read_only = True, 
    data_only = True, 
    IndexPos = None, 
    ColumnPos = None, 
    PrintMessage = False, 
) : 
    wb = openpyxl.load_workbook(ofp,read_only=read_only,data_only=data_only)
    res = list()
    for sn in wb.sheetnames : 
        df = pandas.DataFrame(wb[sn].values)
        if all(k is None for k in [IndexPos,ColumnPos]) : 
            df_21 = df
        else : 
            try : 
                df_21 = df.drop(columns=IndexPos, index=ColumnPos)
            except : 
                if PrintMessage : 
                    print('failed in taking columns (from IndexPos="',IndexPos,'") and/or index (from ColumnPos="',ColumnPos,'") in',sn)
                df_21 = df
            else : 
                if ColumnPos is not None : 
                    if IndexPos is not None : 
                        df_21.columns = df.iloc[ColumnPos,:].drop(IndexPos)
                    else : 
                        df_21.columns = df.iloc[ColumnPos,:]
                if IndexPos is not None : 
                    if ColumnPos is not None : 
                        df_21.index = df.iloc[:,IndexPos].drop(ColumnPos)
                    else : 
                        df_21.index = df.iloc[:,IndexPos]
        res.append((sn,df_21))
    else : 
        ddf = dict(res)
    return(ddf)



# SKLEARN 統計処理関係

class Handle_OneHotEncoder: 
    '''
    # Dousa ok, Usage ; 
    cl = Handle_OneHotEncoder()
    # 反転付きのOne-Hot
    out = cl.fittransform(df.loc[:,'target(name)'].values.reshape(-1, 1)).out(AddExtendInv=True)
    display(out.df_1H)
    # 通常のOne-Hot
    out = cl.fittransform(df.loc[:,'target(name)'].values.reshape(-1, 1)).out()
    display(out.df_1H)
    '''
    def __init__(self, *, npa_fit=None) : 
        self.handle_unknown = 'ignore'
        self.enc = preprocessing.OneHotEncoder(handle_unknown=self.handle_unknown) #"ignore"にするの大事
        if npa_fit is not None : 
            self.enc.fit(npa_fit)
        
    def fit(self, npa_fit) : 
        self.enc.fit(npa_fit)
        return self
    def transform(self, npa_transform) : 
        self.OHE_out = self.enc.transform(npa_transform)
        return self
    def fittransform(self, npa_fit) : 
        self.enc.fit(npa_fit)
        self.OHE_out = self.enc.transform(npa_fit)
        return self
        
    def out(self, *, AddExtendInv=False, np_astype='int64', AddExtendInv_axis=-1) : 
        npa_1Ha = self.OHE_out.toarray().astype(np_astype)
        if AddExtendInv : 
            npa_1Hb = numpy.logical_not(self.OHE_out.toarray()).astype(np_astype)
            self.npa_1H = numpy.stack([npa_1Ha,npa_1Hb], axis=AddExtendInv_axis)
            self.df_1H = pandas.concat([
                pandas.DataFrame(npa_1Ha, columns=self.enc.categories_), 
                pandas.DataFrame(npa_1Hb, columns=self.enc.categories_),
            ], axis=1, keys=['original','inverted'])
        else : 
            self.npa_1H = npa_1Ha
            self.df_1H = pandas.DataFrame(npa_1Ha, columns=self.enc.categories_)
        return self
    


def ShowROC(fpr, tpr, auc_value, *, comment='') : 
    '''
    '''
    lw = 2
    if len(comment)>0 : 
        label = 'AUC = '+str(round(auc_value,2))+' ('+comment+')'
    else : 
        # label = 'ROC curve (area = %0.2f)' % auc_value
        label = 'AUC = '+str(round(auc_value,2))
    plt.figure()
    plt.plot(fpr, tpr, lw=lw, label=label)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.gca().set_aspect('equal', adjustable='box') 
    plt.show()
    return
    


def PlotHistory(
    *, 
    train_loss=None, 
    val_loss=None, 
    train_accu=None, 
    val_accu=None, 
    train_epochs=None, 
    val_epochs=None, 
    train_loss_title = 'Loss vs epochs', 
    train_accu_title = 'Accuracy vs epochs', 
) : 
    # "loss" 評価専用；学習過程
    if train_loss is not None : 
        print(train_loss_title)
        if train_epochs is None : 
            train_epochs = [i for i,v in enumerate(train_loss)]
        plt.plot(train_epochs, train_loss, label="train loss")
        if val_loss is not None : 
            if val_epochs is None : 
                val_epochs = [i for i,v in enumerate(val_loss)]
            plt.plot(val_epochs, val_loss, label="val loss")
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
    # "accu" 評価専用；学習過程
    if train_accu is not None : 
        print(train_accu_title)
        if train_epochs is None : 
            train_epochs = [i for i,v in enumerate(train_accu)]
        plt.plot(traiepochs, train_accu, label="train accuracy")
        if val_accu is not None : 
            if val_epochs is None : 
                val_epochs = [i for i,v in enumerate(val_accu)]
            plt.plot(val_epochs, val_accu, label="test accuracy")
        # plt.yscale('log')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
    return
    


def Handle_RocAucScore_AccuracyScore(TrueLabels, PredictProba, *, PredictedLabels=None, Threshold=0.5, ShowROC_InLine=False, comment='') : 
    fpr, tpr, thr = sklearn.metrics.roc_curve(TrueLabels, PredictProba) #  # 入力順注意 https://nishiohirokazu.hatenadiary.org/entry/20150601/1433165478
    auc = sklearn.metrics.auc(fpr, tpr) # 
    if PredictedLabels is None : 
        PredictedLabels = (PredictProba > Threshold).astype(int)
    length = len(TrueLabels)
    # auc = roc_auc_score(TrueLabels, PredictProba) # 同じ値が得られるが、この関数を使用しない
    acc = sklearn.metrics.accuracy_score(TrueLabels, PredictedLabels) # 
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(TrueLabels, PredictedLabels, average='binary', zero_division=0) # 
    l = [length, auc, acc, precision, recall, f1]
    d = {'length':length, 'auc':auc, 'acc':acc, 'precision':precision, 'recall':recall, 'f1':f1}
    if ShowROC_InLine : 
        ShowROC(fpr, tpr, auc, comment=comment )
    return(d)


# Pubchem

def Get_AutoSimilarities(
    ps, *, 
    LO_function_ForFingerprint = None, # LO_function_ForFingerprint, 
    LO_function_ForSimilarity = DataStructs.BulkTanimotoSimilarity, 
    GetPdistLike = False, 
    DuplicateTriangleMatrix = False, 
    UsePandas = True, 
    Visualize = True, 
) : 
    ''' Usage ; 
    ps = dfp.set_index('pc').loc[:,'smiles'] # pc:PubChemNumber
    df_AutoSim = Get_AutoSimilarities(ps, LO_function_ForFingerprint=LO_function_ForFingerprint, LO_function_ForSimilarity=DataStructs.BulkTanimotoSimilarity)
    SimThr = 0.9
    pc = min(set(flatten_list((i0,i1) for (i0,i1),b in (df_AutoSim > SimThr).stack().items() if b))); print(pc)
    '''
    ll = [[ix,*LO_function_ForFingerprint(v)] for ix,v in ps.items()]
    FP_list = [fp for ix,b,fp in ll if b]
    d_index = dict(enumerate(ix for ix,b,fp in ll if b))
    
    lislis = [(i,LO_function_ForSimilarity(refFP, FP_list[i+1:None])) for i,refFP in enumerate(FP_list)] # triangular matrix; コンビネーションで類似度を計算
    if GetPdistLike : 
        # print( [((i,j),v) for i,l in lislis for j,v in enumerate(l,i+1)] ) # PdistLike並び順の確認
        res = {
            'PDl' : numpy.array([v for i,l in lislis for j,v in enumerate(l,i+1)] ), 
            'd_index' : d_index, 
        }
    else : 
        d = dict(((i,j),v) for i,l in lislis for j,v in enumerate(l,i+1))
        if DuplicateTriangleMatrix : 
            d = dict(((ix0,ix1),d.get((ix0,ix1),d.get((ix1,ix0),None))) for ix0 in range(len(lislis)) for ix1 in range(len(lislis)))
        
        d_rn = dict((tuple(d_index.get(i) for i in t),v) for t,v in d.items())
        if UsePandas : 
            res = pandas.Series(d_rn).unstack() # "autocorrelation" of Tanimoto # df_AutoSim
            if Visualize : 
                plt.figure()
                sns.heatmap(res) # df_AutoSim
                plt.show() # グラフを表示する
        else : 
            res = d_rn 
    return(res) # df_AutoSim


def LO_function_ForFingerprint(smi, *, LO=lambda m:Chem.RDKFingerprint(m) ) : 
    '''
    ps_mol = dfp.loc[:,'smiles'].map(lambda smi:Handle_MolFromSmiles(smi))
    # ps_fps = ps_mol.map(lambda m:None if m is None else AllChem.GetMorganFingerprintAsBitVect(m, 2,2048))
    ps_fps = ps_mol.map(lambda m:None if m is None else AllChem.GetMACCSKeysFingerprint(m))
    # ps_fps = ps_mol.map(lambda m:None if m is None else pyAvalonTools.GetAvalonFP(m))
    # ps_fps = ps_mol.map(lambda m:None if m is None else Chem.RDKFingerprint(m))
    '''
    b,m,fp = True,None,None
    try : 
        m = Chem.MolFromSmiles(smi) 
    except : 
        b,m,fp = False,None,None
    else : 
        try : 
            fp = LO(m)
            # fp = AllChem.GetMorganFingerprintAsBitVect(m, 2,2048)
            # fp = AllChem.GetMACCSKeysFingerprint(m)
            # fp = pyAvalonTools.GetAvalonFP(m)
            # fp = Chem.RDKFingerprint(m)
        except : 
            b,m,fp = False,m,None
        else : 
            pass # "Complete"
    return(b,fp)
    