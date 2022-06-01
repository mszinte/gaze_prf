
import argparse
import os.path as op
import os
from braincoder.models import GaussianPRF2DWithHRF
from braincoder.hrf import CustomHRFModel, SPMHRFModel
from braincoder.optimize import ParameterFitter
from braincoder.utils import get_rsq
from gaze_prf.utils.data import get_data, get_prf_model, get_dm, get_grid_coordinates, get_masker, get_all_roi_labels
from braincoder.optimize import ParameterFitter
import pandas as pd
import numpy as np

from nilearn import image
from itertools import product


def main(subject, bids_folder):

    target_dir = op.join(bids_folder, 'derivatives', 'prf_predictive_accuracy', f'sub-{subject}',
                         'func', )

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    dm = get_dm(bids_folder=bids_folder, task='GazeCenter',
                resize_factor=3.0)
    paradigm = dm.reshape((len(dm), -1))

    masker = get_masker(subject, roi='brainmask',
                        bids_folder=bids_folder)

    for task in ['AttendStim', 'AttendFix']:
        retinotopic_model = get_prf_model(
            subject, task=task, gaze='Center', bids_folder=bids_folder)

        r2s_retinotopic = []
        r2s_spatiotopic = []

        for gaze in ['Left', 'Right']:

            spatiotopic_model = get_prf_model(
                subject, task=task, bids_folder=bids_folder)

            if gaze == 'Left':
                spatiotopic_model.parameters['x'] += 4
            elif gaze == 'Right':
                spatiotopic_model.parameters['x'] -= 4

            data = get_data(subject, bids_folder=bids_folder,
                            fullscreen=False,
                            task=task,
                            gaze=gaze,
                            masker=masker)

            print(data)

            data = data.groupby(['time']).mean()

            # Correct for baseline/amplitude effects
            for model in [retinotopic_model, spatiotopic_model]:
                optimizer = ParameterFitter(model, data, paradigm)
                model.parameters = optimizer.refine_baseline_and_amplitude(
                    model.parameters)

            retinotopic_predictions = retinotopic_model.predict(
                paradigm=paradigm)
            spatiotopic_predictions = spatiotopic_model.predict(
                paradigm=paradigm)

            print(retinotopic_predictions)

            r2_retinotopic = np.clip(
                get_rsq(data, retinotopic_predictions), 0, 1)
            r2_spatiotopic = np.clip(
                get_rsq(data, spatiotopic_predictions), 0, 1)

            r2s_retinotopic.append(r2_retinotopic)
            r2s_spatiotopic.append(r2_spatiotopic)

            gardner = (r2_retinotopic - r2_spatiotopic) / \
                (r2_retinotopic + r2_spatiotopic)

            masker.inverse_transform(r2_retinotopic).to_filename(op.join(target_dir,
                                                                         f'sub-{subject}_task-{task}Gaze{gaze}_desc-retinotopic_r2.nii.gz'))
            masker.inverse_transform(r2_spatiotopic).to_filename(op.join(target_dir,
                                                                         f'sub-{subject}_task-{task}Gaze{gaze}_desc-spatiotopic_r2.nii.gz'))
            masker.inverse_transform(gardner).to_filename(op.join(target_dir,
                                                                  f'sub-{subject}_task-{task}Gaze{gaze}_gardner.nii.gz'))

        r2s_retinotopic = np.mean(r2s_retinotopic, 0)
        r2s_spatiotopic = np.mean(r2s_spatiotopic, 0)
        gardner = (r2s_retinotopic - r2s_spatiotopic) / \
            (r2s_retinotopic + r2s_spatiotopic)

        df = []
        for d, label in zip([r2s_retinotopic, r2s_spatiotopic, gardner], ['desc-retinotopic_r2', 'desc-spatiotopic_r2',
                                                                          'gardner']):
            d = masker.inverse_transform(d)

            d.to_filename(
                op.join(target_dir, f'sub-{subject}_task-{task}_{label}.nii.gz'))
            df.append(d)

        df = image.concat_imgs(df)

        for roi in get_all_roi_labels():
            roi_masker = get_masker(subject, roi, bids_folder, 'both')

            p = roi_masker.fit_transform(df)
            p = pd.DataFrame(
                p.T, columns=['r2_retinotopic', 'r2_spatiotopic', 'gardner'])
            p.index.name = 'voxel'
            p.to_csv(op.join(
                target_dir, f'sub-{subject}_task-{task}_r2s.tsv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument(
        '--bids_folder', default='/tank/shared/2021/visual/pRFgazeMod/')

    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder)
