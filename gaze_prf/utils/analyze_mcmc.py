import arviz as az
import numpy as np
from gaze_prf.utils.data import get_dm_parameters
from braincoder.barstimuli import get_angle_radius_from_xy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pingouin

def get_hdi(d_in):
    d_out = pd.DataFrame(az.hdi(d_in.values.ravel(), .95)).T
    d_out.columns = ['ci_min', 'ci_max']

    return d_out.loc[0]

def get_mcmc_summary(barpars, n_burnin, task='FS', dropna=True):

    barpars = barpars[barpars['sample'] > n_burnin-1]
    if 'y' in barpars.columns:
        barpars = get_angle_radius_from_xy(barpars)

    barpars.columns.name = 'parameter'
    barpars = barpars.set_index(['frame', 'chain', 'sample']).sort_index().stack('parameter').to_frame('value')

    rhat = barpars.unstack('chain').groupby(['frame', 'parameter']).apply(lambda x:
            az.rhat(x.values.T)).to_frame('rhat')

    map = barpars.groupby(['frame', 'parameter']).mean().rename(columns={'value':'estimate'})
    hdi = barpars.groupby(['frame', 'parameter']).apply(get_hdi)
    std = barpars.groupby(['frame', 'parameter']).std().rename(columns={'value':'std'})
    ground_truth = get_dm_parameters(task=task)
    ground_truth = ground_truth.assign(direction=ground_truth['angle'].map({0.0:'horizontal', .5*np.pi:'vertical'}))
    ground_truth.set_index('direction', append=True, inplace=True)
    ground_truth = ground_truth.stack(dropna=dropna).to_frame('value')

    pars = ground_truth.join(map).join(hdi).join(std).join(rhat).reset_index('direction')

    pars.loc[:, 'error'] = pars['estimate'] - pars['value']
    pars.loc[:, 'abs(error)'] = pars['error'].abs()
    pars.loc[:, 'ci_width'] = pars['ci_max'] - pars['ci_min']


    return pars


def plot_mcmc_summary(summary_pars, nan_empty_frames=True, task='FS',  **kwargs):

    # if nan_empty_frames:
        # ground_truth = get_dm_parameters(task=task).stack(dropna=False).to_frame('value')

    # if task == 'aperture':
        # ground_truth = ground_truth.query('parameter in ["x", "height"]')

    # summary_pars = ground_truth.join(summary_pars.drop('value', axis=1).reset_index().set_index(['frame', 'parameter']),
            # on=['frame', 'parameter'])

    fac = sns.FacetGrid(summary_pars.reset_index(), col='parameter', col_wrap=3, sharey=False, aspect=2.,
            **kwargs)

    fac.map(plt.plot, 'frame', 'value', color='k')
    fac.map(plt.plot, 'frame', 'estimate')
    fac.map(plt.fill_between, 'frame', 'ci_min', 'ci_max', alpha=.5)


def check_prediction(summary_pars, by_direction=True):
    
    if by_direction:
        return summary_pars.groupby(['parameter', 'direction']).apply(lambda d: pingouin.corr(d['value'], d['estimate']))
    else:
        return summary_pars.groupby('parameter').apply(lambda d: pingouin.corr(d['value'], d['estimate']))


def error_calibration(summary_pars, by_direction=True):
    
    if by_direction:
        return summary_pars.groupby(['parameter', 'direction']).apply(lambda d: pingouin.corr(d['ci_width'], d['abs(error)']))
    else:
        return summary_pars.groupby('parameter').apply(lambda d: pingouin.corr(d['ci_width'], d['abs(error)']))
