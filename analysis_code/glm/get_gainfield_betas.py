import os
import os.path as op
import pandas as pd
import numpy as np
import argparse
from gaze_prf.utils.data import get_prf_parameters, get_masker, get_dm, get_grid_coordinates, get_data
from braincoder.models import GaussianPRF2DWithHRF
from braincoder.hrf import SPMHRFModel
from braincoder.utils import get_rsq
from tqdm.contrib.itertools import product
from tensorflow.linalg import lstsq

def main(subject, bids_folder='/tank/shared/2021/visual/pRFgazeMod', resize_factor=3):

    target_dir = op.join(bids_folder, 'derivatives', 'gain_field_betas', f'sub-{subject}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)


    dm = get_dm(bids_folder=bids_folder, resize_factor=resize_factor,
            task='GazeCenter').astype(float)

    dm /= resize_factor**2
    grid_coordinates = get_grid_coordinates(
        bids_folder=bids_folder, resize_factor=resize_factor)
    paradigm = dm.reshape((len(dm), -1))
    hrf_model = SPMHRFModel(1.317025)

    prf_pars = []
    keys = []
    for task, gaze in product(['AttendFix', 'AttendStim'], ['Center']):
        prf_pars.append(get_prf_parameters(subject, None, gaze, task, None, None, None, False))
        keys.append((task, gaze))

    prf_pars = pd.concat(prf_pars, keys=keys, names=['task', 'gaze'])

    model = GaussianPRF2DWithHRF(
        grid_coordinates, paradigm, hrf_model=hrf_model)

    pred = model.predict(parameters=prf_pars)

    data = []

    keys = []
    for task, gaze in product(['AttendFix', 'AttendStim'], ['Left', 'Center', 'Right']):
        data.append(get_data(subject, None, gaze, task))
        keys.append((task, gaze)) 
    
    data = pd.concat(data)

    data = data.groupby(['task', 'gaze', 'time', ]).mean()


    masker = get_masker(subject)

    im = op.join(bids_folder, f'pp_data/sub-{subject}/func/fmriprep_dct', f'sub-{subject}_ses-01_task-AttendStimGazeCenterFS_run-1_fmriprep_dct.nii.gz')
    masker.fit(im)

    for task, target in product(['AttendStim', 'AttendFix'], ['Left', 'Right']):
        # n_voxels x n_timespoints x 2
        X = pred[(task, 'Center')].values.T
        X = np.stack((np.ones_like(X), X), axis=2)
        Y = data.loc[(task, target)].values.T[..., np.newaxis]
        beta = lstsq(X, Y, fast=False).numpy()

        beta_im = masker.inverse_transform(beta[:, 1, 0])

        beta_im.to_filename(op.join(target_dir, f'sub-{subject}_task-{task}StimGaze{target}_beta.nii.gz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument(
        '--bids_folder', default='/tank/shared/2021/visual/pRFgazeMod/')

    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder)
