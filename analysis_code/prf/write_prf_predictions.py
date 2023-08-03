import numpy as np
import pandas as pd
import argparse
import os.path as op
import os
from braincoder.models import GaussianPRF2DWithHRF
from braincoder.hrf import CustomHRFModel, SPMHRFModel
from braincoder.utils import get_rsq
from braincoder.optimize import ParameterFitter
from gaze_prf.utils.data import get_data, get_prf_parameters, get_dm, get_grid_coordinates, get_masker, get_all_roi_labels


def main(subject, bids_folder, resize_factor=3.):

    target_dir = op.join(bids_folder, 'derivatives', 'prf_fits.mean', f'sub-{subject}',
                         'func', )

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    # ALLL DATA
    masker = get_masker(subject, roi='brainmask',
            bids_folder=bids_folder)

    data = get_data(subject, bids_folder=bids_folder,
            task=None, session=None,
            # task='AttendStim', session='02',
            masker=masker)
    fs_pars = get_prf_parameters(subject, session=None, task=None, run=None,
            masker=masker,
            bids_folder=bids_folder)

    dm = get_dm(bids_folder=bids_folder, resize_factor=resize_factor)/resize_factor**2.
    grid_coordinates = get_grid_coordinates(
        bids_folder=bids_folder, resize_factor=resize_factor)
    paradigm = dm.reshape((len(dm), -1))
    hrf_model = SPMHRFModel(1.317025)

    model = GaussianPRF2DWithHRF(
        grid_coordinates, paradigm, parameters=fs_pars, hrf_model=hrf_model)

    predictions = model.predict()

    fn = op.join(target_dir, f'sub-{subject}_desc-gaussprf_predictions.nii.gz')
    masker.inverse_transform(predictions).to_filename(fn)

    fn = op.join(target_dir, f'sub-{subject}_desc-mean_bold.nii.gz')
    masker.inverse_transform(data.groupby('time').mean()).to_filename(fn)

    target_dir = op.join(bids_folder, 'derivatives', 'prf_fits', f'sub-{subject}',
                         'func', )

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    for task, d in data.groupby(['task']):

        pars = get_prf_parameters(subject, session=None, task=task, run=None, masker=masker,
                bids_folder=bids_folder)
        model = GaussianPRF2DWithHRF(
            grid_coordinates, paradigm, parameters=pars, hrf_model=hrf_model)

        predictions = model.predict()

        fn = op.join(target_dir, f'sub-{subject}_task-{task}_desc-gaussprf_predictions.nii.gz')
        masker.inverse_transform(predictions).to_filename(fn)

        fn = op.join(target_dir, f'sub-{subject}_task-{task}_desc-mean_bold.nii.gz')
        masker.inverse_transform(d.groupby('time').mean()).to_filename(fn)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('subject', default=None)

    parser.add_argument(
        '--bids_folder', default='/tank/shared/2021/visual/pRFgazeMod/')

    args=parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder)
