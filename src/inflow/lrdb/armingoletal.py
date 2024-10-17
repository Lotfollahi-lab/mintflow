

import os, sys
import numpy as np
import pandas as pd
import re
import math


class LRPair:
    def __init__(self, group_ligname: list, group_ligID: list, group_recname: list, group_recID: list, confidence: str):
        self.group_ligname = group_ligname
        self.group_ligID = group_ligID
        self.group_recname = group_recname
        self.group_recID = group_recID
        self.confidence = confidence

        for ell in [group_ligname, group_ligID, group_recname, group_recID]:
            assert isinstance(ell, list)
            for u in ell:
                assert isinstance(u, str)
                try:
                    assert u[0] != ' '
                    assert u[-1] != ' '
                except:
                    print("u = {}".format(u))

        assert isinstance(confidence, str)

    def __str__(self):
        toret = ''
        for attr_name in ['group_ligname', 'group_ligID', 'group_recname', 'group_recID', 'confidence']:
            toret = toret + "{}: {}\n".format(attr_name, getattr(self, attr_name))

        return toret

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def listLRPair_to_df(input_listLRPair):
        '''
        Converts a list of `LRPair`-s to a df.
        '''
        assert isinstance(input_listLRPair, list)
        for u in input_listLRPair:
            assert isinstance(u, LRPair)

        pd_data, pd_colname = [], []
        for lrobj in input_listLRPair:
            lrobj: LRPair
            curr_pd_data, curr_pd_colname = [], []
            for attrname in ["group_ligname", "group_ligID", "group_recname", "group_recID", "confidence"]:
                curr_pd_data.append(
                    "__".join(getattr(lrobj, attrname)) if (attrname != "confidence") else getattr(lrobj, attrname)
                )
                curr_pd_colname.append(
                    attrname
                )

            pd_data.append(curr_pd_data)
            pd_colname.append(curr_pd_colname)

        return pd.DataFrame(
            data=pd_data,
            columns=pd_colname[0]
        )


str_NA = "N.A."


def func_getLR_Qiao(path_files):
    df = pd.read_excel(
        os.path.join(
            path_files,
            'Human-2014-Qiao-LR-pairs.xlsx'
        )
    )
    return [
        LRPair(
            group_ligname=[str(df.iloc[n]['Ligand (Symbol)'])],
            group_ligID=[str(df.iloc[n]['Ligand      (Entrez ID)'])],
            group_recname=[str(df.iloc[n]['Receptor (Symbols)'])],
            group_recID=[str(df.iloc[n]['Receptor (Entrez ID)'])],
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
            group_ligname=[df.iloc[n]['Ligand']],
            group_ligID=[str_NA],
            group_recname=[df.iloc[n]['Receptor']],
            group_recID=[str_NA],
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
            group_ligname=[df.iloc[n]['Ligand.ApprovedSymbol']],
            group_ligID=[str_NA],
            group_recname=[df.iloc[n]['Receptor.ApprovedSymbol']],
            group_recID=[str_NA],
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
            group_ligname=[df.iloc[n]['AliasA']],
            group_ligID=[df.iloc[n]['uidA']],
            group_recname=[df.iloc[n]['AliasB']],
            group_recID=[df.iloc[n]['uidB']],
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
            group_ligname=[df.iloc[n]['ligand']],
            group_ligID=[str_NA],
            group_recname=[df.iloc[n]['receptor']],
            group_recID=[str_NA],
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
            group_ligname=[df.iloc[n]['From']],
            group_ligID=[str_NA],
            group_recname=[df.iloc[n]['To']],
            group_recID=[str_NA],
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
            group_ligname=[df.iloc[n]['Ligand gene symbol']],
            group_ligID=[df.iloc[n]['Ligand HGNC ID']],
            group_recname=[df.iloc[n]['Receptor gene symbol']],
            group_recID=[df.iloc[n]['Receptor HGNC ID']],
            confidence=str_NA
        )
        for n in range(df.shape[0])
    ]


def drop_beginend_space(str_input):
    assert isinstance(str_input, str)
    toret = str_input + ""
    if toret[0] == ' ':
        toret = toret[1:]
    if toret[-1] == ' ':
        toret = toret[0:-1]
    return toret


def fun_getLR_Kirouac(path_files):
    df = pd.read_excel(
        os.path.join(
            path_files,
            'Human-2010-Kirouac-LR-pairs.xlsx'
        )
    )
    '''
    The df is a bit messy.
    In the ligands column
        - most of them are like "explanation (genename)"
        - some of them are like "genename (genename)", where inside/outside of () can have some slashes.
        - clues to drop explanations
            - explanations are lower case
            - they usually have spaces and or commas.
            - ..
        - processing assumptions:
            - Ligands column:
                - there are two parts, outside paranthesis and inside paranthesis
                - each part is split by slashes
                - each chunk is now decided to be either genename or expalanation
                    - has space (not in the beginning or ending positions) --> explanation
                    - has comma --> explanation
                    - all lower case --> explanation
                    - otherwise --> genename
            - Receptors column:
                - / means after slash words are concatenated.
                - '-' means a range of numbers, like 1-4.
                - some even have paranthesis.
                - If there are any /-s or '-'-s, there are two parts
                    - before /,-
                    - after /,-
                - Assumptions
                    - if there is no '/', '-', and '()' --> a single genename
                    - if there is '()' --> no '/' and '-'
                    - otherwise tehrea re '-'s or '/'-s.



    '''

    def _process_ligstr(ligstr):
        before_par = ligstr[0:ligstr.find("(")]
        after_par = ligstr[ligstr.find('(') + 1:ligstr.find(')')]
        list_toret = []
        for u_ in [before_par, after_par]:
            u = u_ + ""

            for v in u.split('/'):
                v = drop_beginend_space(v)

                flag_is_explanation = False
                if (' ' in v) and ('/' not in v):
                    flag_is_explanation = True
                if (',' in v) and ('/' not in v):
                    flag_is_explanation = True
                if v.islower():
                    flag_is_explanation = True

                if not flag_is_explanation:
                    list_toret.append(v)

        return list_toret

    def _process_recstr(recstr):
        # but there are 'XXXX 1A/1B/1C etc --> so its not only digits slash'
        flag_has_DDDdashDDD = re.search('[0-9]+-[0-9]+', recstr) is not None

        if (re.search('[0-9]+-[0-9]+', recstr) is not None) and (drop_beginend_space(recstr) not in ['4-1BB']):
            list_toret = []
            before_dash = recstr[0:re.search('[0-9]+-[0-9]+', recstr).start() - 1]
            after_dash = recstr[re.search('[0-9]+-[0-9]+', recstr).start():]
            before_dash = drop_beginend_space(before_dash)
            after_dash = drop_beginend_space(after_dash)
            # for u in before_dash:
            min_rng = min([int(u) for u in after_dash.split('-')])
            max_rng = max([int(u) for u in after_dash.split('-')])
            for v in range(min_rng, max_rng + 1):
                list_toret.append("{} {}".format(before_dash, str(v)))
            list_toret = [drop_beginend_space(u) for u in list_toret]
            return list_toret

        elif '/' in recstr:
            list_toret = []
            before_dash = recstr[0:re.search(' .+/', recstr).start()]
            after_dash = recstr[re.search(' .+/', recstr).start():]
            before_dash = drop_beginend_space(before_dash)
            after_dash = drop_beginend_space(after_dash)
            # for u in before_dash:
            for v in after_dash.split('/'):
                if (re.match('[0-9]+.*', v) is not None) or (v in ['A', 'B']):
                    list_toret.append("{} {}".format(before_dash, v))
                else:
                    list_toret.append(v)
            list_toret = [drop_beginend_space(u) for u in list_toret]
            return list_toret

        elif '(' in recstr:
            assert (')' in recstr)
            outside_par = recstr[0:recstr.find('(') - 1]
            inside_par = recstr[recstr.find('(') + 1: recstr.find(')') - 1]

            # drop the starting/ending spaces
            inside_par = drop_beginend_space(inside_par)
            outside_par = drop_beginend_space(outside_par)

            return [drop_beginend_space(inside_par), drop_beginend_space(outside_par)]
        else:
            return [drop_beginend_space(recstr)]

    return [
        LRPair(
            group_ligname=_process_ligstr(df.iloc[n]['LIGAND']),
            group_ligID=[str_NA],
            group_recname=_process_recstr(df.iloc[n]['RECEPTOR(S)']),
            group_recID=[str_NA],
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
    def _is_nan(strorfloat_in):
        if isinstance(strorfloat_in, str):
            if set(strorfloat_in) == set([' ']):
                return True
            else:
                return False
        else:
            return math.isnan(strorfloat_in)

    df = pd.read_excel(
        os.path.join(
            path_files,
            'Human-2020-Noël-LR-pairs.xlsx'
        )
    )
    list_toret = []
    for n in range(df.shape[0]):
        list_Lig = [drop_beginend_space(str(df.iloc[n]['Ligand 1']))]
        if not _is_nan(df.iloc[n]['Ligand 2']):
            list_Lig.append(
                drop_beginend_space(str(df.iloc[n]['Ligand 2']))
            )

        list_Rec = [drop_beginend_space(str(df.iloc[n]['Receptor 1']))]
        if not _is_nan(df.iloc[n]['Receptor 2']):
            list_Rec.append(
                drop_beginend_space(str(df.iloc[n]['Receptor 2']))
            )
        if not _is_nan(df.iloc[n]['Receptor 3']):
            list_Rec.append(
                drop_beginend_space(str(df.iloc[n]['Receptor 3']))
            )

        list_toret.append(
            LRPair(
                group_ligname=list_Lig,
                group_ligID=[str_NA],
                group_recname=list_Rec,
                group_recID=[str_NA],
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

    list_toret = []
    for n in range(df.shape[0]):
        list_Ligname = [str(df.iloc[n]['Ligand.ApprovedSymbol'])]
        list_Recname = []
        if isinstance(df.iloc[n]['Receptor.ApprovedSymbol'], str):
            list_Recname.append(
                df.iloc[n]['Receptor.ApprovedSymbol']
            )
        else:
            assert math.isnan(
                df.iloc[n]['Receptor.ApprovedSymbol']
            )
            list_Recname.append(str_NA)

        list_toret.append(
            LRPair(
                group_ligname=list_Ligname,
                group_ligID=[str_NA],
                group_recname=list_Recname,
                group_recID=[str_NA],
                confidence=str_NA
            )
        )

    return list_toret


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

    list_toret = []
    for n in range(df.shape[0]):
        list_lig = [drop_beginend_space(df.iloc[n]['ligand_symbol'])]
        list_rec = [drop_beginend_space(str(u)) for u in str(df.iloc[n]['receptor_symbol']).split('&')]
        list_toret.append(
            LRPair(
                group_ligname=list_lig,
                group_ligID=[str_NA],
                group_recname=list_rec,
                group_recID=[str_NA],
                confidence=str(df.iloc[n]['evidence'].count(';'))
            )
        )  # TODO: add the ensemble IDs

    return list_toret


def fun_getLR_Shao(path_files):
    '''
    TODO: in this file each pair contains one ligand and one receptor, but ligands are repeated across different pairs.
    Should we group them?
    '''
    df = pd.read_csv(
        os.path.join(
            path_files,
            'Human-2020-Shao-LR-pairs.txt'
        ),
        delimiter='\t'
    )
    list_toret = []
    for n in range(df.shape[0]):
        list_toret.append(
            LRPair(
                group_ligname=[str(df.iloc[n]['ligand_gene_symbol'])],
                group_ligID=[str(df.iloc[n]['ligand_ensembl_gene_id'])],
                group_recname=[str(df.iloc[n]['receptor_gene_symbol'])],
                group_recID=[str(df.iloc[n]['receptor_ensembl_gene_id'])],
                confidence=str(df.iloc[n]['evidence'])
            )
        )
    return list_toret


def func_getLR_Zheng(path_files):
    raise NotImplementedError("The df doesn't seem to have matching L-R pairs?")


def fun_getLR_Dimitrov(path_files):
    '''
    The "resource" column is entirely consensus.
    In both 'source_genesymbol' and 'target_genesymbol' columns there are complexes separated by underscore.
    '''

    df = pd.read_csv(
        os.path.join(
            path_files,
            'Human-2022-Dimitrov-LR-pairs.csv'
        )
    )
    list_toret = []
    for n in range(df.shape[0]):
        list_toret.append(
            LRPair(
                group_ligname=[drop_beginend_space(u) for u in str(df.iloc[n]['source_genesymbol']).split('_')],
                group_ligID=[str_NA],
                group_recname=[drop_beginend_space(u) for u in str(df.iloc[n]['target_genesymbol']).split('_')],
                group_recID=[str_NA],
                confidence=str(df.iloc[n]['resource'])
            )
        )

    return list_toret


def func_getLR_Vento(path_files):
    raise NotImplementedError(
        "partner_b column contains both ligand and receptor gene names? So skipped this file."
    )


def fun_getLR_Omnipath(path_files):
    '''
    The is_directed column is entirely 1.

    '''
    df = pd.read_csv(
        os.path.join(
            path_files,
            'Human-2021-OmniPath-Turei',
            'OmniPathPPIs.tsv'
        ),
        delimiter='\t'
    )
    df = df[
        (df['consensus_stimulation'] == 1) & (df['consensus_inhibition'] == 0) & (df['consensus_direction'] == 1)
        ]  # only the stimulations for which there is consensus (not to have L-R pairs with confusing direction)

    list_toret = []
    for n in range(df.shape[0]):
        list_toret.append(
            LRPair(
                group_ligname=[drop_beginend_space(u) for u in
                               str(df.iloc[n]['source']).replace("COMPLEX:", "").split('_')],
                group_ligID=[str_NA],
                group_recname=[drop_beginend_space(u) for u in
                               str(df.iloc[n]['target']).replace("COMPLEX:", "").split('_')],
                group_recID=[str_NA],
                confidence=str_NA
            )
        )

    return list_toret


def fun_getLR_NicheNet(path_files):
    '''
    TODO: in this file each pair contains one ligand and one receptor, but ligands are repeated across different pairs.
    Should we group them?
    - In neither 'from' column nor the 'to' column there is not underscore or complex (they look single genenames).
    '''
    df = pd.read_csv(
        os.path.join(
            path_files,
            'Human-2019-Browaeys-LR-pairs',
            'NicheNet-LR-pairs.csv'
        )
    )
    list_toret = []
    for n in range(df.shape[0]):
        list_toret.append(
            LRPair(
                group_ligname=[str(df.iloc[n]['from'])],
                group_ligID=[str_NA],
                group_recname=[str(df.iloc[n]['to'])],
                group_recID=[str_NA],
                confidence=str_NA
            )
        )

    return list_toret


