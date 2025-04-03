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
    is_outlier_c = is_outlier.copy()
    scores_sol = {}
    for sol, c in zip(solutions, is_outlier_c.columns):
        precision, recall, f1 = scores(sol, is_outlier_c.index[is_outlier[c] == 1])
        scores_sol[c] = pd.DataFrame({'precision': precision, 'recall': recall, 'f1': f1}, index=[0])
    scores_sol = pd.concat(scores_sol, axis=0)
    return scores_sol

def plot_solutions(df, solutions, title):
    fig, ax = plt.subplots(3, 1, figsize=(20, 5),layout='tight')
    for i, s in enumerate(solutions):
        df.iloc[:, i].plot(ax=ax[i])
        ax[i].plot(s, df.iloc[:, i].iloc[s], 'r.')
    plt.suptitle(title, x=0.01, horizontalalignment='left')

# ------------------------------- MargniIppolitoLoddo ----------------------------------------------------------------
path_1 = join(sol_data_dir, 'MargniIppolitoLoddo.pk')
with open(path_1, 'rb') as f:
    sol_1 = pk.load(f)

scores_mil = get_scores([s.index for s in sol_1])


# ------------------------------- ColomboWeyBerchtold ----------------------------------------------------------------

solutions_2 = [pd.read_csv(join(sol_data_dir, n), index_col=0).index for n in ['outliers_1.csv', 'outliers_2.csv', 'outliers_3.csv']]
scores_cwb = get_scores(solutions_2)


# ------------------------------- PalaGrigioniGubeli ----------------------------------------------------------------
with open(join(sol_data_dir, 'PalaGrigioniGubeli.pk'), 'rb') as f:
    solutions_pala = pk.load(f)
scores_pala = get_scores(solutions_pala)


res = pd.concat({'mil':scores_mil, 'cwb':scores_cwb, 'pgg':scores_pala}, axis=1)


plot_solutions(data, solutions_pala, 'scores_2, f1: t0={:0.2f}, t1={:0.2f}, t2={:0.2f}'.format(*scores_pala['f1'].values))


# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# ------------------------------- 2024 -------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# Manuel
new_sol_data_dir = '/home/lorenzo/Documents/Teaching/outliers_groupworks/2024/'
solutions_1 = pd.read_csv(join(new_sol_data_dir, "manuel_outliers.csv"))
solutions_1 = [np.argwhere(s).ravel() for s in solutions_1[['outlier_1','outlier_2','outlier_3']].values.T]
scores_1 = get_scores(solutions_1)

plot_solutions(data, solutions_1, 'scores_1, f1: t0={:0.2f}, t1={:0.2f}, t2={:0.2f}'.format(*scores_1['f1'].values))

# Stefano
solutions_2 = pd.read_csv(join(new_sol_data_dir, "stefano/comprehensive_outliers_detection.csv"))
solutions_2 = [np.argwhere(s).ravel() for s in solutions_2[['0_is_outlier','1_is_outlier','2_is_outlier']].values.T]
scores_2 = get_scores(solutions_2)

plot_solutions(data, solutions_2, 'scores_2, f1: t0={:0.2f}, t1={:0.2f}, t2={:0.2f}'.format(*scores_2['f1'].values))


# Passoni Farina
solutions_3 = pd.read_csv(join(new_sol_data_dir, "outliers_Passoni_Farina.csv"))
solutions_3 = [np.argwhere(s).ravel() for s in solutions_3[['0','1','2']].values.T]
scores_3 = get_scores(solutions_3)

plot_solutions(data, solutions_3, 'scores_3, f1: t0={:0.2f}, t1={:0.2f}, t2={:0.2f}'.format(*scores_3['f1'].values))


# Vairani
solutions_4 = pd.read_csv(join(new_sol_data_dir, "vairani_outliers.csv"))
solutions_4 = [np.argwhere(s).ravel() for s in solutions_4[['saliency_0','saliency_1','saliency_2']].values.T]
scores_4 = get_scores(solutions_4)

plot_solutions(data, solutions_4, 'scores_3, f1: t0={:0.2f}, t1={:0.2f}, t2={:0.2f}'.format(*scores_4['f1'].values))


# My solution
my_solution = pd.read_pickle("TimeSeriesForecasting/data/solution_outliers.pk")
my_solution = [np.argwhere(s).ravel() for s in my_solution.values.T]
my_scores = get_scores(my_solution)

plot_solutions(data, my_solution, 'my scores, f1: t0={:0.2f}, t1={:0.2f}, t2={:0.2f}'.format(*scores_1['f1'].values))


res_2024 = pd.concat([scores_1['f1'].rename('Acquistapace'), scores_3['f1'].rename('Passoni-Farina'), scores_2['f1'].rename('Billeter'), scores_4['f1'].rename('Vairani')], axis=1)

res_2024.plot(kind='bar', figsize=(8, 4), rot=0)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.savefig('TimeSeriesForecasting/figs/out_1.png')
plt.close('all')

res_2024.mean().plot(kind='bar', rot=0, figsize=(8, 4))
plt.gca().spines[['top', 'right']].set_visible(False)
plt.savefig('TimeSeriesForecasting/figs/out_2.png')
plt.close('all')

res_2024 = pd.concat([scores_1['f1'].rename('Acquistapace'), scores_3['f1'].rename('Passoni-Farina')], axis=1)
pd.concat([my_scores['f1'].rename('LorenzLorenz'), res_2024], axis=1).mean().plot(kind='bar', rot=0, figsize=(8, 4))
plt.gca().spines[['top', 'right']].set_visible(False)
plt.savefig('TimeSeriesForecasting/figs/out_3.png')
plt.close('all')

pd.concat([scores_pala['f1'].rename('PalaGrigioniGubeli'), my_scores['f1'].rename('LorenzLorenz'), res_2024], axis=1).mean().plot(kind='bar', rot=0, figsize=(8, 4))
plt.gca().spines[['top', 'right']].set_visible(False)
plt.savefig('TimeSeriesForecasting/figs/out_4.png')
plt.close('all')

pd.concat([scores_pala['f1'].rename('PalaGrigioniGubeli'), my_scores['f1'].rename('LorenzLorenz'), res_2024], axis=1).mean().plot(kind='bar', rot=0, figsize=(8, 4))
plt.hlines(scores_pala['f1'].rename('PalaGrigioniGubeli').mean(), plt.gca().get_xlim()[0]+0.25, 4.25, linewidth=1)
plt.hlines(1, *plt.gca().get_xlim(), linewidth=1, color='red', linestyle='--')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.savefig('TimeSeriesForecasting/figs/out_5.png')
plt.close('all')




# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# ------------------------------- 2025 -------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
from glob import glob
new_sol_data_dir = '/home/lorenzo/Documents/Teaching/outliers_groupworks/2025/'
solutions_paths = glob(join(new_sol_data_dir, '*'))

def df_to_list(df):
    return [df[c].dropna().astype(int).values for c in df.columns]

sols = {}
for s in solutions_paths:
    extension = s.split('.')[-1]
    if extension == 'csv':
        df = pd.read_csv(s)
    elif extension == 'pkl':
        res = pd.read_pickle(s)
        if isinstance(res, pd.DataFrame):
            df = pd.concat(res, axis=1)
        else:
            df = pd.concat([pd.Series(s) for s in res], axis=1)
    elif extension == 'json':
        df = pd.read_json(s)
    if df.shape[0]<df.shape[1]:
        df = df.T
    df.columns = [0, 1, 2]
    solutions = df_to_list(df)
    print(s.split('/')[-1])
    print([s.sum() for s in solutions])
    sols[(s.split('/')[-1]).split('.')[0]] = solutions

fig, ax = plt.subplots(3, 1, figsize=(20, 5), layout='tight')
for i in range(3):
    pd.concat([pd.Series(v[i], name=k).astype(int) for k, v in sols.items()], axis=1).plot(ax=ax[i])

sols_2023 = {'pala_grigioni_gubeli':solutions_pala, 'lorenzlorenz':my_solution}
sols_all = {**(sols_2023), **sols}

fig, ax = plt.subplots(3, 1, figsize=(20, 5))
for i in range(3):
    pd.concat([pd.Series(v[i], name=k).astype(int) for k, v in sols_all.items()], axis=1).plot(ax=ax[i])

scores_f1 = pd.concat({k:get_scores(v)['f1'] for k, v in sols_all.items()}, axis=1)
scores_f1['best_2024'] = res_2024['Acquistapace']
scores_f1.mean().sort_values().plot(kind='bar', figsize=(8, 4), rot=0)
plt.semilogy()
plt.show()

for k, v in sols.items():
    plot_solutions(data, v, 'scores_3, f1: t0={:0.2f}, t1={:0.2f}, t2={:0.2f}, {}'.format(*scores_f1[k].values, k.upper()))
    plt.savefig('TimeSeriesForecasting/figs/outliers_2025/{}.png'.format(k))