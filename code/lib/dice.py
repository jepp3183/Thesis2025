import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.datasets import load_iris, load_wine

# Sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# DiCE imports
import dice_ml
from dice_ml.utils import helpers  # helper functions

import lib.baseline as baseline
import lib.util as util
from lib.baycon import baycon_explainer

from lib.eval.eval_metrics import *

def dice_explainer(total_CFs=10):
    def exp(X, classifier, initial_point, target_label):
        df = pd.DataFrame(X)
        df['y'] = classifier.predict(X)
        d = dice_ml.Data(dataframe=df, continuous_features=list(range(X.shape[1])), outcome_name='y')
        m = dice_ml.Model(model=classifier, backend='sklearn')
        exp = dice_ml.Dice(d, m)

        X_df = df.drop('y', axis=1)
        e = exp.generate_counterfactuals(X_df[initial_point:initial_point+1], total_CFs=total_CFs, desired_class=target_label)
        return e.cf_examples_list[0].final_cfs_df.drop('y', axis=1).to_numpy()
    
    return exp