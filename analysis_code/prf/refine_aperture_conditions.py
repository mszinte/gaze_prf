import argparse
import os.path as op
import os
from braincoder.models import GaussianPRF2DWithHRF
from braincoder.hrf import CustomHRFModel, SPMHRFModel
from braincoder.optimize import ParameterFitter
from braincoder.utils import get_rsq
from gaze_prf.utils.data import get_data, get_prf_parameters, get_dm, get_grid_coordinates, get_masker, get_all_roi_labels

import pandas as pd
import numpy as np

from nilearn import image
from itertools import product


def main(subject, bids_folder, starting_model='retinotopic',
         task=None, gaze=None, resize_factor=3):

    target_dir = op.join(bids_folder, 'derivatives', 'prf_fits', f'sub-{subject}',
                         'func', )

    assert(starting_model in ['retinotopic', 'spatiotopic', 'combine'])

    if starting_model == 'combine':

        for task, gaze in product(['AttendStim', 'AttendFix'], ['Left', 'Right', 'Center']):
            r2_retinotopic = image.load_img(op.join(target_dir,
                                                    f'sub-{subject}_task-{task}Gaze{gaze}_desc-gaussprf.startwith-retinotopic.r2_parameters.nii.gz'))
            r2_spatiotopic = image.load_img(op.join(target_dir,
                                                    f'sub-{subject}_task-{task}Gaze{gaze}_desc-gaussprf.startwith-spatiotopic.r2_parameters.nii.gz'))

            mask_retinotopic = image.math_img('r2_retinotopic >= r2_spatiotopic', r2_retinotopic=r2_retinotopic,
                                              r2_spatiotopic=r2_spatiotopic)

            parameter_labels = ['x', 'y', 'sd', 'amplitude', 'baseline', 'ecc', 'theta', 'r2']
            parameter_images = []

            for parameter_label in parameter_labels:
                par_retinotopic = op.join(target_dir,
                                          f'sub-{subject}_task-{task}Gaze{gaze}_desc-gaussprf.startwith-retinotopic.{parameter_label}_parameters.nii.gz')
                par_spatiotopic = op.join(target_dir,
                                          f'sub-{subject}_task-{task}Gaze{gaze}_desc-gaussprf.startwith-spatiotopic.{parameter_label}_parameters.nii.gz')

                par_combined = image.math_img('np.where(mask_retinotopic, par_retinotopic, par_spatiotopic)',
                        par_retinotopic=par_retinotopic,
                        par_spatiotopic=par_spatiotopic,
                        mask_retinotopic=mask_retinotopic)

                parameter_images.append(par_combined)

                new_fn = op.join(target_dir, f'sub-{subject}_task-{task}Gaze{gaze}_desc-gaussprf.optim.{parameter_label}_parameters.nii.gz')
                par_combined.to_filename(new_fn)

            pars_im = image.concat_imgs(parameter_images)

            for roi in get_all_roi_labels():
                roi_masker = get_masker(subject, roi, bids_folder, 'both')
                p = roi_masker.fit_transform(pars_im)
                p = pd.DataFrame(p.T, columns=parameter_labels)
                p.index.name = 'voxel'
                p.to_csv(op.join(
                    target_dir, f'sub-{subject}_task-{task}Gaze{gaze}_roi-{roi}_desc-gaussprf.optim_parameters.tsv'))
    else:
        if not op.exists(target_dir):
            os.makedirs(target_dir)

        masker = get_masker(subject, roi='brainmask',
                            bids_folder=bids_folder)

        data = get_data(subject, bids_folder=bids_folder,
                        fullscreen=False,
                        task=task,
                        gaze=gaze,
                        masker=masker)
        chunks = data.columns // 50000
        data.columns = pd.MultiIndex.from_arrays(
            [chunks, data.columns], names=['chunk', 'voxel'])

        fs_pars = get_prf_parameters(subject, bids_folder=bids_folder)
        fs_pars.index = data.columns

        print(fs_pars)

        dm = get_dm(bids_folder=bids_folder, resize_factor=resize_factor,
                    task='GazeCenter')
        grid_coordinates = get_grid_coordinates(
            bids_folder=bids_folder, resize_factor=resize_factor)
        paradigm = dm.reshape((len(dm), -1))
        hrf_model = SPMHRFModel(1.317025)

        def fit_prf(d, init_pars):

            model = GaussianPRF2DWithHRF(
                grid_coordinates, paradigm, data=d, hrf_model=hrf_model)

            optimizer = ParameterFitter(model, model.data, paradigm)

            pars = optimizer.refine_baseline_and_amplitude(init_pars)
            pars = optimizer.fit(init_pars=pars, learning_rate=1e-3, max_n_iterations=10000,
                                 fixed_pars=['y'],
                                 r2_atol=0.0001)

            pars = optimizer.refine_baseline_and_amplitude(pars)
            pred = model.predict(parameters=pars)

            r2 = get_rsq(model.data, pred)

            pars['r2'] = r2
            pars['ecc'] = np.sqrt(pars['x']**2 + pars['y']**2)
            pars['theta'] = np.angle(pars['x'] + pars['y'] * 1j)

            print(pars.describe())

            return pars.T

        def get_init_pars(ix, starting_model, gaze):

            pars = fs_pars.loc[ix].copy()

            if starting_model == 'spatiotopic':
                if gaze == 'Left':
                    pars['x'] += 4
                elif gaze == 'Right':
                    pars['x'] -= 4

            return pars

        data = data.groupby(['task', 'gaze', 'time']).mean()

        for (task, gaze), d in data.groupby(['task', 'gaze']):
            print(task, gaze, d)

            pars = []

            for chunk, d_ in d.groupby('chunk', 1):
                print(f"Working on task {task} chunk {chunk}")
                init_pars = get_init_pars(d_.columns, starting_model, gaze)
                pars.append(fit_prf(d_, init_pars))

            pars = pd.concat(pars, 1).T
            print(pars.describe())

            pars_im = masker.inverse_transform(pars.T)

            for parameter in ['x', 'y', 'sd', 'amplitude', 'baseline', 'ecc', 'theta', 'r2']:
                masker.inverse_transform(pars[parameter]).to_filename(op.join(target_dir,
                                                                              f'sub-{subject}_task-{task}Gaze{gaze}_desc-gaussprf.startwith-{starting_model}.{parameter}_parameters.nii.gz'))

            pars_im = masker.inverse_transform(pars.T)

            for roi in get_all_roi_labels():
                roi_masker = get_masker(subject, roi, bids_folder, 'both')
                p = roi_masker.fit_transform(pars_im)
                p = pd.DataFrame(p.T, columns=pars.columns)
                p.index.name = 'voxel'
                p.to_csv(op.join(
                    target_dir, f'sub-{subject}_task-{task}Gaze{gaze}_roi-{roi}_desc-gaussprf.startwith-{starting_model}_parameters.tsv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('starting_model')
    parser.add_argument('--gaze', default=None)
    parser.add_argument('--task', default=None)

    parser.add_argument(
        '--bids_folder', default='/tank/shared/2021/visual/pRFgazeMod/')

    args = parser.parse_args()

    main(args.subject, starting_model=args.starting_model,
         bids_folder=args.bids_folder, gaze=args.gaze, task=args.task)
