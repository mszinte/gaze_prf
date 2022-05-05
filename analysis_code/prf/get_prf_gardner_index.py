import argparse
from gaze_prf.utils.data import get_data, get_prf_parameters, get_dm, get_grid_coordinates, get_masker, get_all_roi_labels
from itertools import product
import os
import os.path as op
import numpy as np
import pandas as pd


def main(subject, bids_folder):

    prf_dir = op.join(bids_folder, 'derivatives', 'prf_fits', f'sub-{subject}',
                      'func', )

    target_dir = op.join(bids_folder, 'derivatives', 'prf_gardner', f'sub-{subject}',
                         'func', )

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    masker = get_masker(subject, roi='brainmask',
                        bids_folder=bids_folder)

    for task in ['AttendStim', 'AttendFix']:
        centergaze_x = masker.fit_transform(op.join(prf_dir,
                                                    f'sub-{subject}_task-{task}GazeCenter_desc-gaussprf.optim.x_parameters.nii.gz'))

        leftgaze_x = masker.fit_transform(op.join(prf_dir,
                                                  f'sub-{subject}_task-{task}GazeLeft_desc-gaussprf.optim.x_parameters.nii.gz'))

        rightgaze_x = masker.fit_transform(op.join(prf_dir,
                                                   f'sub-{subject}_task-{task}GazeRight_desc-gaussprf.optim.x_parameters.nii.gz'))

        ss_retinotopic = (
            (np.stack((leftgaze_x, rightgaze_x), 0) - centergaze_x)**2).sum(0)
        spatiotopic_prediction_left = centergaze_x + 4
        spatiotopic_prediction_right = centergaze_x - 4
        ss_spatiotopic = ((np.stack((leftgaze_x, rightgaze_x), 0) - np.stack((spatiotopic_prediction_left,
                                                                              spatiotopic_prediction_right), 0))**2).sum(0)

        gardner = (ss_spatiotopic - ss_retinotopic) / \
            (ss_spatiotopic + ss_retinotopic)

        gardner = masker.inverse_transform(gardner)

        gardner.to_filename(
            op.join(target_dir,
                f'sub-{subject}_task-{task}_prfgardner.nii.gz'))


        for roi in get_all_roi_labels():
            roi_masker = get_masker(subject, roi, bids_folder, 'both')
            p = roi_masker.fit_transform(gardner)
            p = pd.DataFrame(p.T, columns=['gardner'])
            p.index.name = 'voxel'
            p.to_csv(op.join(
                target_dir, f'sub-{subject}_task-{task}_roi-{roi}_prfgardner.tsv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)

    parser.add_argument(
        '--bids_folder', default='/tank/shared/2021/visual/pRFgazeMod/')

    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder)
