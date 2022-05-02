import argparse
import os.path as op
import os
from braincoder.models import GaussianPRF2DWithHRF
from braincoder.hrf import CustomHRFModel, SPMHRFModel
from braincoder.optimize import ParameterFitter
from gaze_prf.utils.data import get_data, get_prf_parameters, get_dm, get_grid_coordinates, get_masker
import pandas as pd
import numpy as np



def main(subject, bids_folder, fullscreen, resize_factor=3):

    target_dir = op.join(bids_folder, 'derivatives', 'prf_fits', f'sub-{subject}',
                         'func', )

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    masker = get_masker(subject, roi='brainmask',
            bids_folder=bids_folder)

    data = get_data(subject, bids_folder=bids_folder,
            masker=masker)
    chunks = data.columns // 50000
    data.columns = pd.MultiIndex.from_arrays([chunks, data.columns], names=['chunk', 'voxel'])

    get_prf_parameters(subject, session=None, gaze='CenterFS', task=None, run=None,
            fullscreen=True,
            masker=masker,
            bids_folder='/tank/shared/2021/visual/pRFgazeMod')

    fs_pars = get_prf_parameters(subject, bids_folder=bids_folder)
    fs_pars.index = data.columns

    print(fs_pars)

    dm = get_dm(bids_folder=bids_folder, resize_factor=resize_factor)
    grid_coordinates = get_grid_coordinates(bids_folder=bids_folder, resize_factor=resize_factor)
    paradigm = dm.reshape((len(dm), -1))
    hrf_model = SPMHRFModel(1.317025)


    def fit_prf(d, init_pars):

        model = GaussianPRF2DWithHRF(
            grid_coordinates, paradigm, data=d, hrf_model=hrf_model)

        optimizer = ParameterFitter(model, model.data, paradigm)

        pars = optimizer.refine_baseline_and_amplitude(init_pars)
        pars = optimizer.fit(init_pars=pars, learning_rate=1e-3, max_n_iterations=10000,
                r2_atol=0.0001)

        pars = optimizer.refine_baseline_and_amplitude(pars)
        pred = model.predict(parameters=pars)

        r2 = get_rsq(model.data, pred)

        pars['r2'] = r2
        pars['ecc'] = np.sqrt(pars['x']**2 + pars['y']**2)
        pars['theta'] = np.angle(pars['x'] + pars['y'] * 1j)

        print(pars.describe())

        return pars.T


    data = data.groupby(['task', 'time']).mean()

    for task, d in data.groupby(['task']):
        print(task, d)

        pars = []

        for chunk, d_ in d.groupby('chunk', 1):
            print(f"Working on task {task} chunk {chunk}")
            pars.append(fit_prf(d_, fs_pars.loc[d_.columns]))
        

        pars = pd.concat(pars, 1).T
        print(pars.describe())

        for parameter in ['x', 'y', 'sd', 'amplitude', 'baseline', 'ecc', 'theta', 'r2']:
            masker.inverse_transform(pars[parameter]).to_filename(op.join(target_dir,
                f'sub-{subject}_task-{task}_roi-{roi}_desc-gaussprf.{parameter}_parameters.nii.gz'))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--fullscreen', action='store_true')

    parser.add_argument(
        '--bids_folder', default='/tank/shared/2021/visual/pRFgazeMod/')

    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder, fullscreen=args.fullscreen)
