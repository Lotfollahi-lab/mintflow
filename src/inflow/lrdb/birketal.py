

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





