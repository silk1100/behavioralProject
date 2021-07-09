"""
1. A new subject(s) is/are given in form of a csv file with features extracted using FreeSurfer.
2. Each testing subject is going to pass through each behavioral ML model to get the diagnosis of that subject given
that model
3. Aggregation of all decisions to give the final classification for that subject
"""

import pandas as pd
import numpy as np

import constants
from sklearn.pipeline import Pipeline

class BehavioralDiagnosis:
    """
    Create a pipeline from all of the behavioral report models
    Feed to the pipeline the raw data of the testing subjects (extracted features from the FreeSUrfer)
    Get the final diagnosis
    """
    def __init__(self, method='performance_weighted', models_dir: dict=None):
        self.method = method
        self.models_dir = models_dir
        if not self.models_dir:
            pass

    def predict(self, X, method='performance_weighted'):
        pass

