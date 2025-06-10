

import os, sys
import numpy as np
import pandas as pd
import re
import math


str_NA = "not_available"


def drop_beginend_space(str_input):
    assert isinstance(str_input, str)
    toret = str_input + ""
    if toret[0] == ' ':
        toret = toret[1:]
    if toret[-1] == ' ':
        toret = toret[0:-1]
    return toret


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


def func_getLR_omnipath(path_files):
    '''
    Now only stimulation signal (and not inhibition signal) is considered.
    TODO: maybe a better approach?
    '''
    df = pd.read_csv(
        os.path.join(
            path_files,
            'omnipath_lr_network.csv'
        )
    )
    df = df[
        df['consensus_stimulation'] & (df['is_inhibition'] == False) & df['consensus_direction']
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
                confidence=str(df.iloc[n]['curation_effort'])
            )
        )

    return list_toret


def func_getLR_nichenet_human(path_files):
    '''
    The two columns 'from' and 'two' simply contains a single genename.
    '''

    df = pd.read_csv(
        os.path.join(
            path_files,
            'nichenet_lr_network_v2_human.csv'
        )
    )

    list_toret = []
    for n in range(df.shape[0]):
        list_toret.append(
            LRPair(
                group_ligname=[drop_beginend_space(str(df.iloc[n]['from']))],
                group_ligID=[str_NA],
                group_recname=[drop_beginend_space(str(df.iloc[n]['to']))],
                group_recID=[str_NA],
                confidence=str_NA
            )
        )
    return list_toret


