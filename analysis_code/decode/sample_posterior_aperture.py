import dill
import os
import os.path as op
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import argparse
from braincoder.barstimuli import BarStimulusFitter
from braincoder.models import LinearModelWithBaselineHRF, GaussianPRF2DWithHRF
from braincoder.hrf import CustomHRFModel, SPMHRFModel
from braincoder.optimize import ResidualFitter
from braincoder.stimuli import CustomStimulusFitter, SzinteStimulus2

from gaze_prf.utils.data import (get_prf_parameters, get_data, get_masker, get_grid_coordinates, get_dm,
                                 get_fixation_screen, get_dm_parameters)


def main(subject, session, task, gaze, n_voxels=250, resize_factor=3., roi='V1',
    bids_folder='/tank/shared/2021/visual/pRFgazeMod'):


    fs_pars = get_prf_parameters(subject, task=task, bids_folder=bids_folder, filter=True,
            roi=roi)

    print(fs_pars)


    fs_pars = fs_pars.loc[~fs_pars.isnull().any(1)]
    mask = fs_pars['r2'].sort_values(ascending=False).index[:n_voxels]
    fs_pars = fs_pars.loc[mask].astype(np.float32)
    fs_pars['amplitude'] /= resize_factor**2

    if n_voxels > len(mask):
        print(f'ATTENTION: asked for {n_voxels}, but only {len(mask)} adhere to criteria')

    n_voxels = len(mask)

    masker = get_masker(subject=subject, roi=roi, bids_folder=bids_folder)
    fs_data = get_data(subject=subject, task=task, masker=masker, bids_folder=bids_folder)
    fs_data = fs_data.loc[:, mask]
    print(fs_data)

    dm = get_dm(bids_folder=bids_folder, resize_factor=resize_factor)
    grid_coordinates = get_grid_coordinates(bids_folder=bids_folder, resize_factor=resize_factor)
    paradigm = dm.reshape((len(dm), -1))
    hrf_model = SPMHRFModel(1.317025)

    print('paradigm', paradigm, paradigm.shape)

    paradigm = np.tile(paradigm.T, 4).astype(np.float32).T
    print('paradigm', paradigm, paradigm.shape)

    print('data', fs_data)
    print('parameters', fs_pars)

    model = GaussianPRF2DWithHRF(grid_coordinates, paradigm, fs_data, fs_pars[['x', 'y', 'sd', 'baseline', 'amplitude']],
                                     hrf_model=hrf_model)

    print(f'Size coordinates: {grid_coordinates.shape}')
    resfitter = ResidualFitter(model, model.data.astype(np.float32))
    omega, dof = resfitter.fit(method='t', max_n_iterations=10000, init_dof=36., init_sigma2=7., init_rho=1e-3)

    test_data = get_data(subject=subject, session=session, task=task, run=1, gaze=gaze, masker=masker, bids_folder=bids_folder)
    test_data = test_data.loc[:, mask].reset_index(['session', 'session', 'task', 'run'], drop=True)
    print('test_data', test_data)

    baseline_image = get_fixation_screen(bids_folder=bids_folder, resize_factor=resize_factor)
    ground_truth = get_dm_parameters(task='aperture', bids_folder=bids_folder)
    relevant_frames = np.where(~ground_truth.isnull().any(1))[0]
    bar_width = ground_truth.loc[relevant_frames].width.mean()
    max_bar_height = grid_coordinates[:, 1].max() - grid_coordinates[:, 1].min()
    print(f'max bar height: {max_bar_height}')

    stimulus = SzinteStimulus2(grid_coordinates, bar_width=bar_width, max_height=max_bar_height, intensity=255.0)
    stimfitter = CustomStimulusFitter(test_data.astype(np.float32), model, stimulus, omega, dof, )

    grid_pars = stimfitter.fit_grid({'x':np.linspace(-8, 8, 10, True, dtype=np.float32),
                            'height':np.linspace(1e-12, 8. - 1e-3, 10,True, dtype=np.float32)}).astype(np.float32)

    fit_pars = stimfitter.fit(grid_pars, learning_rate=0.0001, fit_vars=['x', 'height'],
            relevant_frames=relevant_frames)

    target_dir = op.join(bids_folder, 'derivatives', 'barfits', f'sub-{subject}', f'ses-{session}',
            'func')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    fit_pars.to_csv(op.join(target_dir,
        f'sub-{subject}_ses-{session}_task-{task}Gaze{gaze}_run-1_roi-{roi}_nvoxels-{n_voxels}_resizefactor-{resize_factor}_barpars.tsv'), sep='\t')

    start = timer()
    samples, stats = stimfitter.sample_posterior(fit_pars, 4, relevant_frames, step_size=1., n_burnin=250,

            n_samples=500, target_accept_prob=.3, falloff_speed=100)
    duration = timer() - start
    stats['elapsed_time'] = duration

    samples.to_csv(op.join(target_dir,
        f'sub-{subject}_ses-{session}_task-{task}Gaze{gaze}_run-1_roi-{roi}_nvoxels-{n_voxels}_resizefactor-{resize_factor}_mcmcbarpars.tsv'), sep='\t')

    with open(op.join(target_dir,
        f'sub-{subject}_ses-{session}_task-{task}Gaze{gaze}_run-1_roi-{roi}_nvoxels-{n_voxels}_resizefactor-{resize_factor}_mcmcstats.pkl'), 'wb') as handle:
        dill.dump(stats, handle)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('task', default=None)
    parser.add_argument('gaze', default=None)

    parser.add_argument(
        '--roi', default='V1')
    parser.add_argument(
        '--resize_factor', default=3, type=int)
    parser.add_argument(
        '--bids_folder', default='/tank/shared/2021/visual/pRFgazeMod/')
    parser.add_argument(
        '--diagonal_covariance', action='store_true')
    parser.add_argument('--n_voxels', default=100, type=int)

    args = parser.parse_args()

    main(args.subject, session=args.session, task=args.task, gaze=args.gaze, n_voxels=args.n_voxels,
         resize_factor=args.resize_factor,
         roi=args.roi,
         bids_folder=args.bids_folder)
