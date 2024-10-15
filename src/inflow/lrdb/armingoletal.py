

import os, sys
import numpy as np
import pandas as pd
from . import str_NA

class LRPair:
    def __init__(self, Lname, Lid, Rname, Rid, confidence):
        self.Lname = Lname
        self.Lid = Lid
        self.Rname = Rname
        self.Rid = Rid
        self.confidence = confidence

        for attr in [self.Lname, self.Lid, self.Rname, self.Rid, self.confidence]:
            assert isinstance(attr, str)

    def __str__(self):
        toret = ''
        for attr_name in ['Lname', 'Lid', 'Rname', 'Rid', 'confidence']:
            toret = toret + "{}: {}\n".format(attr_name, getattr(self, attr_name))

        return toret

    def __repr__(self):
        return self.__str__()


def func_getLR_Qiao(path_files):
    df = pd.read_excel(
        os.path.join(
            path_files,
            'Human-2014-Qiao-LR-pairs.xlsx'
        )
    )
    return [
        LRPair(
            Lname=str(df.iloc[n]['Ligand (Symbol)']),
            Lid=str(df.iloc[n]['Ligand      (Entrez ID)']),
            Rname=str(df.iloc[n]['Receptor (Symbols)']),
            Rid=str(df.iloc[n]['Receptor (Entrez ID)']),
            confidence=str(df.iloc[n]['Confidence from iRefWeb'])
        )
        for n in range(df.shape[0])
    ]

def func_getLR_Pavlicev(path_files):
    df = pd.read_excel(
        os.path.join(
            path_files,
            'Human-2017-Pavlicev-LR-pairs.xlsx'
        )
    )
    return [
        LRPair(
            Lname=df.iloc[n]['Ligand'],
            Lid=str_NA,
            Rname=df.iloc[n]['Receptor'],
            Rid=str_NA,
            confidence=str_NA
        )
        for n in range(df.shape[0])
    ]

def func_getLR_Ramilowski(path_files):
    df = pd.read_csv(
        os.path.join(
            path_files,
            'Human-2015-Ramilowski-LR-pairs.txt'
        ),
        delimiter='\t'
    )
    return [
        LRPair(
            Lname=df.iloc[n]['Ligand.ApprovedSymbol'],
            Lid=str_NA,
            Rname=df.iloc[n]['Receptor.ApprovedSymbol'],
            Rid=str_NA,
            confidence=str(df.iloc[n]['Pair.Evidence'])
        )
        for n in range(df.shape[0])
    ]



def func_getLR_Ximerakis(path_files):
    df = pd.read_csv(
        os.path.join(
            path_files,
            'Human-2019-Ximerakis-BaderLab-2017.txt'
        ),
        delimiter='\t'
    )
    return [
        LRPair(
            Lname=df.iloc[n]['AliasA'],
            Lid=df.iloc[n]['uidA'],
            Rname=df.iloc[n]['AliasB'],
            Rid=df.iloc[n]['uidB'],
            confidence=str(df.iloc[n]['confidence'])
        )
        for n in range(df.shape[0])
    ]


def func_getLR_Cabello(path_files):
    df = pd.read_csv(
        os.path.join(
            path_files,
            'Human-2020-Cabello-Aguilar-LR-pairs.csv'
        )
    )
    return [
        LRPair(
            Lname=df.iloc[n]['ligand'],
            Lid=str_NA,
            Rname=df.iloc[n]['receptor'],
            Rid=str_NA,
            confidence=str(len(df.iloc[n]['source'].split(',')))
        )
        for n in range(df.shape[0])
    ]

def fun_getLR_Choi(path_files):
    df = pd.read_csv(
        os.path.join(
            path_files,
            'Human-2015-Choi-LR-pairs.txt'
        ),
        delimiter='\t'
    )
    return [
        LRPair(
            Lname=df.iloc[n]['From'],
            Lid=str_NA,
            Rname=df.iloc[n]['To'],
            Rid=str_NA,
            confidence=str_NA
        )
        for n in range(df.shape[0])
    ]

def fun_getLR_Hou(path_files):
    df = pd.read_excel(
        os.path.join(
            path_files,
            'Human-2020-Hou-LR-pairs.xlsx'
        )
    )
    return [
        LRPair(
            Lname=df.iloc[n]['Ligand gene symbol'],
            Lid=df.iloc[n]['Ligand HGNC ID'],
            Rname=df.iloc[n]['Receptor gene symbol'],
            Rid=df.iloc[n]['Receptor HGNC ID'],
            confidence=str_NA
        )
        for n in range(df.shape[0])
    ]


def fun_getLR_Kirouac(path_files):
    df = pd.read_excel(
        os.path.join(
            path_files,
            'Human-2010-Kirouac-LR-pairs.xlsx'
        )
    )
    return [
        LRPair(
            Lname=df.iloc[n]['LIGAND'],
            Lid=str_NA,
            Rname=df.iloc[n]['RECEPTOR(S)'],  # TODO: ask: this one has 'RECEPTOR(S)' instead of a single receptor.
            Rid=str_NA,
            confidence=str_NA
        )
        for n in range(df.shape[0])
    ]


def fun_getLR_Zhao(path_files):
    df = pd.read_csv(
        os.path.join(
            path_files,
            'Human-2023-Zhao-LR-pairs.tsv'
        ),
        delimiter='\t'
    )
    raise NotImplementedError("There seems to be a groupping between ligands and receptors --> not implemented yet.")
    return [
        LRPair(
            Lname=df.iloc[n]['LIGAND'],
            Lid=str_NA,
            Rname=df.iloc[n]['RECEPTOR(S)'],  # TODO: ask: this one has 'RECEPTOR(S)' instead of a single receptor.
            Rid=str_NA,
            confidence=str_NA
        )
        for n in range(df.shape[0])
    ]


def fun_getLR_Noel(path_files):
    df = pd.read_excel(
        os.path.join(
            path_files,
            'Human-2020-Noël-LR-pairs.xlsx'
        )
    )
    list_toret = []
    for n in range(df.shape[0]):
        L1, L2, R1, R2, R3 = str(df.iloc[n]['Ligand 1']), str(df.iloc[n]['Ligand 2']), \
            str(df.iloc[n]['Receptor 1']), str(df.iloc[n]['Receptor 2']), str(df.iloc[n]['Receptor 3'])

        for l in [L1, L2]:
            for r in [R1, R2, R3]:
                if l != 'nan':
                    if r != 'nan':
                        # Now each pair of a group of L-R-s form a separate pair -->
                        # TODO: handle the grouping later on.
                        list_toret.append(
                            LRPair(
                                Lname=l,
                                Lid=str_NA,
                                Rname=r,
                                Rid=str_NA,
                                confidence=str_NA
                            )
                        )

    return list_toret


def fun_getLR_Wang(path_files):
    '''
    TODO:NOTE interestingly many of the lists (including this one) seem to have orphan ligands.
    In which case Receptor.ApprovedSymbol is 'nan'.
    '''
    df = pd.read_csv(
        os.path.join(
            path_files,
            "Human-2019-Wang-LR-pairs.csv"
        )
    )
    return [
        LRPair(
            Lname=str(df.iloc[n]['Ligand.ApprovedSymbol']),
            Lid=str_NA,
            Rname=str(df.iloc[n]['Receptor.ApprovedSymbol']),
            Rid=str_NA,
            confidence=str_NA
        )
        for n in range(df.shape[0])
    ]



def fun_getLR_Jin(path_files):
    '''
    There are orpahn ligands
    There are usually a pair of receptors.
    '''
    df = pd.read_csv(
        os.path.join(
            path_files,
            'Human-2020-Jin-LR-pairs.csv'
        )
    )

    list_LR = []
    for n in range(df.shape[0]):
        l = df.iloc[n]['ligand_symbol']
        for idx_r, r in enumerate(str(df.iloc[n]['receptor_symbol']).split('&')):
            list_LR.append(
                LRPair(
                    Lname=str(l),
                    Lid=str(df.iloc[n]['ligand_ensembl']),
                    Rname=str(r),
                    Rid=str(df.iloc[n]['receptor_ensembl']).split('&')[idx_r],
                    confidence=str_NA
                )
            )


    return list_LR



def fun_getLR_Shao(path_files):
    df = pd.read_csv(
        os.path.join(
            path_files,
            'Human-2020-Shao-LR-pairs.txt'
        ),
        delimiter='\t'
    )
    return [
        LRPair(
            Lname=str(df.iloc[n]['ligand_gene_symbol']),
            Lid=str(df.iloc[n]['ligand_ensembl_gene_id']),
            Rname=str(df.iloc[n]['receptor_gene_symbol']),
            Rid=str(df.iloc[n]['receptor_ensembl_gene_id']),
            confidence=str(df.iloc[n]['evidence'])
        )
        for n in range(df.shape[0])
    ]


def func_getLR_Zheng(path_files):
    raise NotImplementedError("The df doesn't seem to have matching L-R pairs?")

def fun_getLR_Dimitrov(path_files):
    '''
    The "resource" column is entirely consensus.
    '''
    df = pd.read_csv(
        os.path.join(
            path_files,
            'Human-2022-Dimitrov-LR-pairs.csv'
        )
    )
    return [
        LRPair(
            Lname=str(df.iloc[n]['source_genesymbol']),
            Lid=str_NA,
            Rname=str(df.iloc[n]['target_genesymbol']),
            Rid=str_NA,
            confidence=str(df.iloc[n]['resource'])
        )
        for n in range(df.shape[0])
    ]

def func_getLR_Vento(path_files):
    raise NotImplementedError(
        "partner_b column contains both ligand and receptor gene names? So skipped this file."
    )


def fun_getLR_Omnipath(path_files):
    df = pd.read_csv(
        os.path.join(
            path_files,
            'Human-2021-OmniPath-Turei',
            'OmniPathPPIs.tsv'
        ),
        delimiter='\t'
    )
    df = df[
        (df['consensus_stimulation' ]==1) & (df['consensus_inhibition' ]==0) & (df['consensus_direction' ]==1)
        ]  # only the stimulations for which there is consensus (not to have L-R pairs with confusing direction)

    return [
        LRPair(
            Lname=str(df.iloc[n]['source']),
            Lid=str_NA,
            Rname=str(df.iloc[n]['target']),
            Rid=str_NA,
            confidence=str_NA
        )
        for n in range(df.shape[0])
    ]


def fun_getLR_NicheNet(path_files):
    df = pd.read_csv(
        os.path.join(
            path_files,
            'Human-2019-Browaeys-LR-pairs',
            'NicheNet-LR-pairs.csv'
        )
    )
    return [
        LRPair(
            Lname=str(df.iloc[n]['from']),
            Lid=str_NA,
            Rname=str(df.iloc[n]['to']),
            Rid=str_NA,
            confidence=str_NA
        )
        for n in range(df.shape[0])
    ]


dict_fname_to_funct = {
    'Human-2014-Qiao-LR-pairs.xlsx':func_getLR_Qiao,
    'Human-2017-Pavlicev-LR-pairs.xlsx':func_getLR_Pavlicev,
    'Human-2015-Ramilowski-LR-pairs.txt':func_getLR_Ramilowski,
    'Human-2015-Choi-LR-pairs.txt':fun_getLR_Choi,
    'Human-2020-Hou-LR-pairs.xlsx':fun_getLR_Hou,
    'Human-2010-Kirouac-LR-pairs.xlsx':fun_getLR_Kirouac,
    'Human-2020-Noël-LR-pairs.xlsx':fun_getLR_Noel,
    'Human-2019-Wang-LR-pairs.csv':fun_getLR_Wang,
    'Human-2020-Jin-LR-pairs.csv':fun_getLR_Jin,
    'Human-2020-Shao-LR-pairs.txt':fun_getLR_Shao,
    'Human-2022-Dimitrov-LR-pairs.csv':fun_getLR_Dimitrov,
    'Human-2019-Ximerakis-BaderLab-2017.txt':func_getLR_Ximerakis,
    'Human-2020-Cabello-Aguilar-LR-pairs.csv':func_getLR_Cabello,
    'Human-2021-OmniPath-Turei/OmniPathPPIs.tsv':fun_getLR_Omnipath,
    'Human-2019-Browaeys-LR-pairs/NicheNet-LR-pairs.csv':fun_getLR_NicheNet
}

