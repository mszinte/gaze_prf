import os
import os.path as op
import argparse
from braincoder.utils import get_rsq
from braincoder.models import GaussianPRF2DAngleWithHRF
from braincoder.hrf import CustomHRFModel, SPMHRFModel
from braincoder.optimize import ParameterFitter
from gaze_prf.utils.data import get_dm, get_grid_coordinates, get_prfpy_hrf, get_data, get_masker, get_all_roi_labels
from nilearn import image
from nilearn.input_data import NiftiMasker
import matplotlib.pyplot as plt
import numpy as np
from tqdm.contrib.itertools import product
import pandas as pd
import numpy as np


def main(subject, bids_folder, normalize='zscore', resize_factor=3):


    masker = get_masker(subject, roi='brainmask',
            bids_folder=bids_folder)

    data = get_data(subject, bids_folder=bids_folder, masker=masker, normalize=normalize)
    mean_data = data.groupby('frame').mean()

    chunks = mean_data.columns // 50000

    mean_data.columns = pd.MultiIndex.from_arrays([chunks, mean_data.columns], names=['chunk', 'voxel'])
    print(mean_data)

    mean_data = mean_data
    print(mean_data)


    dm = get_dm(bids_folder=bids_folder, resize_factor=resize_factor)
    grid_coordinates = get_grid_coordinates(bids_folder=bids_folder, resize_factor=resize_factor)
    paradigm = dm.reshape((len(dm), -1))
    hrf_model = SPMHRFModel(1.317025)

    def fit_prf(d):
        print(d)

        model = GaussianPRF2DAngleWithHRF(
            grid_coordinates, paradigm, data=d, hrf_model=hrf_model)

        optimizer = ParameterFitter(model, model.data, paradigm)


        theta = np.linspace(-np.pi, np.pi, 50)
        ecc = np.linspace(.1, 1, 50, True)**2 * 15
        sd = np.linspace(.25, 1., 50).astype(np.float32)**2 * 20
        baseline = [0.0]
        amplitude = [0.00001]

        grid_pars = optimizer.fit_grid(theta, ecc, sd, baseline, amplitude, use_correlation_cost=True)
        refined_grid_pars = optimizer.refine_baseline_and_amplitude(grid_pars)
        refined_grid_r2 = get_rsq(model.data, model.predict(parameters=refined_grid_pars))

        pars = optimizer.fit(init_pars=refined_grid_pars, learning_rate=1e-4, max_n_iterations=10000,
                min_n_iterations=1000,
                r2_atol=0.0001)


        refined_pars = optimizer.refine_baseline_and_amplitude(pars)
        pred = model.predict(parameters=refined_pars)

        refined_r2 = get_rsq(model.data, pred)

        refined_pars['r2'] = refined_r2
        # refined_pars['ecc'] = np.sqrt(refined_pars['x']**2 + refined_pars['y']**2)
        # refined_pars['theta'] = np.angle(refined_pars['x'] + refined_pars['y'] * 1j)
        refined_pars['x'] = np.cos(refined_pars['theta']) * refined_pars['ecc']
        refined_pars['y'] = np.sin(refined_pars['theta']) * refined_pars['ecc']

        print(refined_pars.describe())

        return refined_pars.T

    pars = mean_data.groupby('chunk', axis=1).apply(fit_prf).T


    print(pars.describe())

    key = 'prf_fits.mean'

    if normalize == 'psc':
        key += '.psc'

    target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func', )


    if not op.exists(target_dir):
        os.makedirs(target_dir)

    for parameter in ['x', 'y', 'sd', 'amplitude', 'baseline', 'ecc', 'theta', 'r2']:
        masker.inverse_transform(pars[parameter]).to_filename(op.join(target_dir,
            f'sub-{subject}_desc-gaussprf.{parameter}_parameters.nii.gz'))

    pars_im = masker.inverse_transform(pars.T)

    for roi in get_all_roi_labels():
        roi_masker = get_masker(subject, roi, bids_folder, 'both')
        p = roi_masker.fit_transform(pars_im)
        p = pd.DataFrame(p.T, columns=pars.columns)
        p.index.name = 'voxel'
        p.to_csv(op.join(target_dir,
        f'sub-{subject}_roi-{roi}_desc-gaussprf.fit_parameters.tsv'), sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)

    parser.add_argument(
        '--bids_folder', default='/tank/shared/2021/visual/pRFgazeMod/')
    parser.add_argument('--normalize', default='zscore')

    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder, normalize=args.normalize)
