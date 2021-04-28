"""
Classes that are responsible for splitting the data according to behavioral report, and block for confounding
variables such as age, gender, and IQ?

The logic behind the implementation of this class is 100% clinical.

Dr. Barnes Email.
"
The SRS T scores have a range: 59 or below is TD/normal. 60-65 is mild, 66-75 is moderate, and above 76 is severe.
Your data looks like that spread holds true.
"
"""
import pandas as pd
import numpy as np
from constants import DATA_DIR, TARGET, ASD, TD


class DataDivisor:
    def __init__(self, data: pd.DataFrame, phenofile: pd.DataFrame, behavioral: str):
        self.df = data

        self.df_pheno = phenofile
        if 'subj_id' in self.df_pheno.columns:
            self.df_pheno.set_index('subj_id', inplace=True)

        self.report_type = behavioral

        self.behavioral_columns_ = []

    def _get_behavioral_columns(self):
        for col in self.df_pheno.columns:
            if self.report_type.lower() in col.lower():
                self.behavioral_columns_.append(col)

    def divide(self, phenotypes:dict=None, behavioral_type:str = None) -> pd.DataFrame:
        """
        phenotype should be a dictionary with keys as phenotype labels e.g (mild, moderate, sever).
        A new column will be added to the data with a label corresponding to each subject based on their
        behavioral_type score. If phenotypes is None, then the predefined phenotypes dict in constants.py is
        used. If behavioral_type is not given, then the total score will be used.
        :param phenotypes: dict
        :param behavioral_type: str
        :return: pd.DataFrame
        """
        self.df_behav = self.df_pheno[self.behavioral_columns_]
        x = 0

if __name__ == '__main__':
    df = pd.read_csv(DATA_DIR['medianMmedianP'], index_col=0)
    df_p = pd.read_csv(DATA_DIR['pheno'], index_col='subj_id')

    divisor = DataDivisor(df, df_p, 'srs')
    df = divisor.divide()


