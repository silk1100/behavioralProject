import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pandas as pd

import warnings; warnings.filterwarnings('ignore')

def setup():
    large=22; med=16; small=12
    params={
        'axes.titlesize': med,
        'legend.fontsize':med,
        'figure.figsize':(16, 10),
        'axes.labelsize': med,
        'xtick.labelsize': med,
        'ytick.labelsize':med,
        'figure.titlesize':large
    }
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")

class Visualizer:
    def __init__(self, df=None, x=None, y=None, hue=None, **kwags) -> None:
        setup()
        self.df = df
        self.x = x
        self.y = y
        self.hue=hue
    
    def parse_kwags(self):
        pass

    def _scatter_df(self):
        pass

    def _scatter_xy(self):
        pass
    
    def scatter_plot(self):
        if self.df is not None:
            pass
        else:
            pass

    def diverging_text(self):
        pass

    def line_plot(self):
        pass


