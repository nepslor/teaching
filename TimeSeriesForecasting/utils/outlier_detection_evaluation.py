import pandas as pd
import numpy as np
from hampel import hampel
import pickle as pk
import matplotlib
matplotlib.use('TkAgg')
from os.path import join
import matplotlib.pyplot as plt
from tqdm import tqdm

sol_data_dir = '/home/lorenzo/Documents/Teaching/outliers_groupworks/'
data = pd.read_csv('TimeSeriesForecasting/data/outliers.csv', index_col=0)
ground_truth = pd.read_csv('TimeSeriesForecasting/data/ground_truth.csv', index_col=0)

is_outlier = data-ground_truth != 0


def scores(outliers, ground_truth):
    precision = len([o for o in outliers if o in ground_truth]) / len(outliers)
    recall = len([o for o in outliers if o in ground_truth]) / len(ground_truth)
    f1 = 2/(1/precision + 1/recall)
    return precision, recall, f1

def get_scores(solutions):
    scores_sol = {}
    for sol, c in zip(solutions, is_outlier.columns):
        precision, recall, f1 = scores(sol, is_outlier.index[is_outlier[c] == 1])
        scores_sol[c] = pd.DataFrame({'precision': precision, 'recall': recall, 'f1': f1}, index=[0])
    scores_sol = pd.concat(scores_sol, axis=0)
    return scores_sol


# ------------------------------- MargniIppolitoLoddo ----------------------------------------------------------------
path_1 = join(sol_data_dir, 'MargniIppolitoLoddo.pk')
with open(path_1, 'rb') as f:
    sol_1 = pk.load(f)

scores_mil = get_scores([s.index for s in sol_1])



# ------------------------------- ColomboWeyBerchtold ----------------------------------------------------------------

solutions_2 = [pd.read_csv(join(sol_data_dir, n), index_col=0).index for n in ['outliers_1.csv', 'outliers_2.csv', 'outliers_3.csv']]
scores_cwb = get_scores(solutions_2)


pd.concat({'mil':scores_mil, 'cwb':scores_cwb}, axis=1)