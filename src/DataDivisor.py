"""
Classes that are responsible for splitting the data according to behavioral report, and block for confounding
variables such as age, gender, and IQ?

The logic behind the implementation of this class is 100% clinical
"""
import pandas as pd
import numpy as np


class DataDivisor:
    def __init__(self, fldr_with_json:str):
        pass

    def based_on_age(self, ranges:list = None):
        pass

    def based_on_gender(self):
        pass

    def based_on_ADOS_G_COMM(self, ranges:list=None):
        pass

    def based_on_ADOS_G_Social(self, ranges:list=None):
        pass

    def based_on_ADOS_G_StereoBehav(self, ranges:list=None):
        pass

    def based_on_ADOS_G_Creativity(self, ranges:list=None):
        pass

    def based_on_ADOS_G_Total(self, ranges:list=None):
        pass

    def based_on_ADOS_2_SOCAFFECT(self, ranges:list=None):
        pass

    def based_on_ADOS_2_RRB(self, ranges:list=None):
        pass