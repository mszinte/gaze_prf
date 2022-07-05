import pandas as pd
import numpy as np
import dill
import matplotlib.pyplot as plt
import glob
import re
import os.path as op
import seaborn as sns
from tqdm import tqdm
import argparse
from joblib import Parallel, delayed
from itertools import product
from gaze_prf.utils.data import get_dm_parameters, get_all_roi_labels
from gaze_prf.utils.analyze_mcmc import get_mcmc_summary

sourcedata = '/tank/shared/2021/visual/pRFgazeMod/'
n_voxels = 250
n_burnin = 250
resize_factor = 3

ground_truth = get_dm_parameters()

def fill_empty_frames(d):
    tmp = pd.DataFrame([], index=ground_truth.index)
    return pd.concat((tmp, d), 1)

stats = []

subjects = [f'{i:03d}' for i in range(1, 9)]
sessions = ['01', '02']
tasks = ['AttendFix', 'AttendStim']
gazes = ['Left', 'Right', 'Center']
runs = [1]

# rois = ['V1']
rois = get_all_roi_labels()

with tqdm(list(product(subjects, sessions, tasks, gazes, runs, rois))) as pbar:
    for subject, session, task, gaze, run, roi in pbar:
        fn = op.join(sourcedata, 'derivatives', 'barfits', f'sub-{subject}', f'ses-{session}', 'func',
                     f'sub-{subject}_ses-{session}_task-{task}Gaze{gaze}_run-1_roi-{roi}_nvoxels-*_resizefactor-{resize_factor}_mcmcstats.pkl')

        if op.exists(fn):
            with open(fn, 'rb') as handle:
                s = dill.load(handle)
                stats.append({'subject': subject,
                              'session': session,
                              'task': task,
                              'run': run,
                              'mask': roi,
                              'n_voxels': n_voxels,
                              'elapsed_time': s['elapsed_time'],
                              'diverged_after_burnin': s['diverging'][n_burnin:].numpy().mean(),
                              'accept_ratio_before_burnin': s['accept_ratio'][:n_burnin].numpy().mean(),
                              'accept_ratio_after_burnin': s['accept_ratio'][n_burnin:].numpy().mean()})

stats = pd.DataFrame(stats)
stats.to_csv(op.join(sourcedata, 'derivatives', 'barfits', f'combined_stats_n_voxels-{n_voxels}.tsv'), sep='\t')

samples = []
summaries = []


def get_sample_summary(subject, session, task, gaze, run, roi):

    fn = op.join(sourcedata, 'derivatives', 'barfits', f'sub-{subject}', f'ses-{session}', 'func',
    f'sub-{subject}_ses-{session}_task-{task}Gaze{gaze}_run-1_roi-{roi}_nvoxels-{n_voxels}_resizefactor-{resize_factor}_mcmcbarpars.tsv')

    if op.exists(fn):
        d = pd.read_csv(fn, sep='\t').rename(columns={'time':'frame'})

        map_ = op.join(sourcedata, 'derivatives', 'barfits', f'sub-{subject}', f'ses-{session}', 'func',
        f'sub-{subject}_ses-{session}_task-{task}Gaze{gaze}_run-1_roi-{roi}_nvoxels-{n_voxels}_resizefactor-{resize_factor}_barpars.tsv')

        map_ = pd.read_csv(map_, index_col=[0], sep='\t')
        map_.index.name = 'frame'
        map_.columns.name = 'parameter'

        summary = get_mcmc_summary(d, n_burnin=n_burnin, task='aperture', dropna=False).query('parameter in ["x", "height"]')
        summary = summary.join(map_.stack().to_frame('map'))
        summary['subject'], summary['session'], summary['task'], summary['gaze'], summary['run'], summary['roi'], summary['n_voxels'] = subject, session, task, gaze, run, roi, n_voxels

        summary  = summary.reset_index().set_index(['subject','session', 'task', 'gaze', 'run', 'roi', 'n_voxels',
            'frame', 'parameter'])
        return summary
    else:
        return pd.DataFrame([])


# s = get_sample_summary('001', '01', 'AttendStim', 1, 'V1')
jobs = []
for subject, session, task, gaze, run, roi in product(subjects, sessions, tasks, gazes, runs, rois):
    for method in ['t.hrf']:
        jobs.append(delayed(get_sample_summary)(subject, session, task, gaze, run, roi))

print('Startting parallel jobs')
summaries = Parallel(n_jobs=32)(jobs)

summaries = pd.concat(summaries)
summaries.to_parquet(op.join(sourcedata, 'derivatives',
             'barfits', f'mcmc_summaries_n_voxels-{n_voxels}.parquet'))
