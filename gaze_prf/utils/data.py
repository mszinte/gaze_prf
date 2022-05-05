from tqdm import tqdm
import itertools
import pandas as pd
import os.path as op
import numpy as np
from scipy import ndimage
from scipy.io import loadmat
from braincoder.models import LinearModelWithBaselineHRF, GaussianPRF2DWithHRF
from braincoder.hrf import CustomHRFModel, SPMHRFModel
from nilearn import image
from nilearn.input_data import NiftiMasker
from itertools import product


def get_dm(bids_folder='/tank/shared/2021/visual/pRFgazeMod/', square=False, resize_factor=1,
           task='GazeCenterFS'):
    dms = loadmat(op.join(bids_folder, 'pp_data', 'visual_dm',
                          f'{task}_vd.mat'))['stim']

    if square:
        pixel_offset = int((dms.shape[0] - dms.shape[1]) / 2)
        new_dms = np.zeros((dms.shape[-1], dms.shape[0], dms.shape[0]))
        for timepoint in range(dms.shape[-1]):
            square_screen = np.zeros_like(new_dms[timepoint])
            square_screen[:, pixel_offset:pixel_offset +
                          dms.shape[1]] = dms[..., timepoint]
            new_dms[timepoint, :, ] = square_screen.T

        dms = new_dms.T

    dms = np.rollaxis(dms, -1, 0)

    if resize_factor != 1:
        dms = np.array([ndimage.zoom(d, 1./resize_factor) for d in dms])
        dms = np.float32(dms)
        dms *= resize_factor**2.

    return dms


def get_grid_coordinates(bids_folder='/tank/shared/2021/visual/pRFgazeMod', flattened=True, resize_factor=1):

    dm = get_dm(bids_folder=bids_folder)
    width_degrees = np.arctan(69/2. / 220) / (2*np.pi) * 360 * 2
    height_degrees = dm.shape[2] / dm.shape[1] * width_degrees

    x_coordinates = np.linspace(-width_degrees/2., width_degrees/2.,
                                dm.shape[1], endpoint=True).astype(np.float32)
    y_coordinates = x_coordinates.copy()[::-1]

    pixel_offset, original_height = 52, 135
    y_coordinates = y_coordinates[pixel_offset:pixel_offset+original_height]

    y, x = np.meshgrid(y_coordinates, x_coordinates)

    x = ndimage.zoom(x, 1./resize_factor)
    y = ndimage.zoom(y, 1./resize_factor)

    if flattened:
        return np.concatenate((x.ravel()[:, np.newaxis],
                               y.ravel()[:, np.newaxis]), 1)
    else:
        return x, y


def get_prfpy_hrf(bids_folder='/tank/shared/2021/visual/pRFgazeMod'):

    return np.loadtxt(op.join(bids_folder, 'derivatives', 'hrf.txt'))


def parse_sets(session, gaze, task, run,
               fullscreen=True):

    if session is None:
        sessions = [f'{x:02d}' for x in range(1, 3)]
    elif type(session) is str:
        sessions = [session]
    else:
        sessions = session

    if gaze is None:
        if fullscreen:
            gazes = ['Center']
        else:
            gazes = ['Left', 'Center', 'Right']
    elif type(gaze) is str:
        gazes = [gaze]
    else:
        gazes = gaze

    if task is None:
        tasks = ['AttendFix', 'AttendStim']
    elif type(task) is str:
        tasks = [task]

    if run is None:
        if fullscreen:
            runs = [1, 2]
        else:
            runs = [1]

    elif (type(run) is int) or (type(run) is str):
        runs = [int(run)]
    else:
        runs = run

    # np.roduct(sessions, gazes, tasks, runs)
    sessions, gazes, tasks, runs = map(
        np.ravel, np.meshgrid(sessions, gazes, tasks, runs))

    labels = [f'{task}Gaze{gaze}' for gaze, task in zip(gazes, tasks)]

    if fullscreen:
        labels = [f'{label}FS' for label in labels]

    return sessions, gazes, tasks, labels, runs


def get_data(subject=None, session=None, gaze=None, task=None, run=None,
             fullscreen=True,
             bids_folder='/tank/shared/2021/visual/pRFgazeMod',
             masker=None):

    if masker is None:
        mask = get_brain_mask(subject, bids_folder)
        masker = NiftiMasker(mask)

    sessions, gazes, tasks, labels, runs = parse_sets(
        session, gaze, task, run, fullscreen)

    keys = []
    data = []

    pbar = tqdm(list(zip(sessions, gazes, tasks, labels, runs)))

    for session, gaze, task, label, run in pbar:
        pbar.set_description(
            f'Session {session}, Gaze {gaze}, Task {task}, run {run}')
        fn = op.join(bids_folder, f'pp_data/sub-{subject}/func/fmriprep_dct',
                     f'sub-{subject}_ses-{session}_task-{label}_run-{run}_fmriprep_dct.nii.gz')
        keys.append((session, gaze, task, run))
        d = pd.DataFrame(masker.fit_transform(fn))
        d.index.name, d.columns.name = 'time', 'voxel'
        data.append(d)

    data = pd.concat(data, keys=keys, names=['session', 'gaze', 'task', 'run'])

    return data


def get_prf_parameters(subject, session=None, gaze=None, task=None, run=None,
                       masker=None,
                       roi=None,
                       bids_folder='/tank/shared/2021/visual/pRFgazeMod'):

    parameters = ['x', 'y', 'sd', 'baseline', 'amplitude', 'r2']

    if (roi is None) and (masker is None):
        mask = get_brain_mask(subject, bids_folder)
        masker = NiftiMasker(mask)

    if (session is None) and (gaze is None) and (task is None) and (run is None):

        fs_pars = []

        if roi is None:

            for parameter in parameters:
                p = op.join(bids_folder, 'derivatives', 'prf_fits.mean', f'sub-{subject}',
                            'func', f'sub-{subject}_desc-gaussprf.{parameter}_parameters.nii.gz')

                p = masker.fit_transform(p).ravel()
                fs_pars.append(pd.DataFrame(
                    {parameter: p}, index=np.arange(len(p))))

            return pd.concat(fs_pars, axis=1)
        else:
            return pd.read_csv(op.join(bids_folder, 'derivatives', 'prf_fits.mean', f'sub-{subject}', 'func',
                                       f'sub-{subject}_roi-{roi}_desc-gaussprf.fit_parameters.tsv'), sep='\t', index_col=0)

    elif (session is None) and (gaze is None) and (task is not None) and (run is None):
        if roi is None:
            pars = []
            for parameter in parameters:
                p = op.join(bids_folder, 'derivatives', 'prf_fits', f'sub-{subject}',
                            'func', f'sub-{subject}_task-{task}_desc-gaussprf.{parameter}_parameters.nii.gz')

                p = masker.fit_transform(p).ravel()
                pars.append(pd.DataFrame(
                    {parameter: p}, index=np.arange(len(p))))

            return pd.concat(pars, axis=1)

        else:
            return pd.read_csv(op.join(bids_folder, 'derivatives', 'prf_fits', f'sub-{subject}', 'func',
                                       f'sub-{subject}_task-{task}_roi-{roi}_desc-gaussprf.fit_parameters.tsv'), sep='\t', index_col=0)

    elif (session is None) and (gaze is not None) and (task is not None) and (run is None):
        if roi is None:
            pars = []
            for parameter in parameters:
                p = op.join(bids_folder, 'derivatives', 'prf_fits', f'sub-{subject}',
                            'func', f'sub-{subject}_task-{task}Gaze{gaze}_desc-gaussprf.optim.{parameter}_parameters.nii.gz')

                p = masker.fit_transform(p).ravel()
                pars.append(pd.DataFrame(
                    {parameter: p}, index=np.arange(len(p))))

            return pd.concat(pars, axis=1)

        else:
            return pd.read_csv(op.join(bids_folder, 'derivatives', 'prf_fits', f'sub-{subject}', 'func',
                                       f'sub-{subject}_task-{task}Gaze{gaze}_roi-{roi}_desc-gaussprf.fit_parameters.tsv'), sep='\t', index_col=0)

    else:
        raise NotImplementedError()


def get_prf_model(subject, session=None, gaze=None, task=None, run=None,
                  masker=None,
                  roi=None,
                  resize_factor=3.,
                  bids_folder='/tank/shared/2021/visual/pRFgazeMod'):

    dm = get_dm(bids_folder=bids_folder, resize_factor=resize_factor,
                task='GazeCenter')
    grid_coordinates = get_grid_coordinates(
        bids_folder=bids_folder, resize_factor=resize_factor)
    paradigm = dm.reshape((len(dm), -1))
    hrf_model = SPMHRFModel(1.317025)

    prf_pars = get_prf_parameters(subject, session, gaze, task, run, masker, roi, bids_folder)

    model = GaussianPRF2DWithHRF(
        grid_coordinates, paradigm, hrf_model=hrf_model,
        parameters=prf_pars)

    return model


def get_brain_mask(subject, bids_folder='/tank/shared/2021/visual/pRFgazeMod/',
                   resample_to_func=True):

    mask = image.load_img(op.join(bids_folder, f'deriv_data/fmriprep/fmriprep/',
                                  f'sub-{subject}', 'anat', f'sub-{subject}_desc-brain_mask.nii.gz'))

    if resample_to_func:
        return resample_mask_to_functional_space(subject, mask, bids_folder)
    else:
        return mask


def get_masker(subject, roi='brainmask',
               bids_folder='/tank/shared/2021/visual/pRFgazeMod/',
               hemi='both'):

    if roi == 'brainmask':
        mask = get_brain_mask(subject, bids_folder, True)
    else:
        mask = get_roi_mask(subject, roi, bids_folder,
                            hemi)

    masker = NiftiMasker(mask)

    return masker


def resample_mask_to_functional_space(subject, mask, bids_folder='/tank/shared/2021/visual/pRFgazeMod/'):
    im = op.join(bids_folder, f'pp_data/sub-{subject}/func/fmriprep_dct',
                 f'sub-{subject}_ses-01_task-AttendStimGazeCenterFS_run-1_fmriprep_dct.nii.gz'),
    mask = image.resample_to_img(mask, im,
                                 interpolation='nearest')
    return mask


def get_roi_mask(subject, mask, bids_folder,
                 hemi='both'):

    if mask not in get_all_roi_labels():
        raise ValueError(f'{mask} does not exist!')

    folder = op.join(bids_folder, 'pp_data', f'sub-{subject}', 'masks')

    if hemi in ['L', 'R']:
        return image.load_img(op.join(folder, f'{mask}_cortical_{hemi}.nii.gz'))
    else:
        mask_l = image.load_img(op.join(folder, f'{mask}_cortical_L.nii.gz'))
        mask_r = image.load_img(op.join(folder, f'{mask}_cortical_R.nii.gz'))
        return image.math_img('mask_l+mask_r', mask_l=mask_l, mask_r=mask_r)


def get_all_roi_labels():
    return ['V1',
            'V2',
            'V3',
            'V3AB',
            'hMT+',
            'iIPS',
            'iPCS',
            'LO',
            'mPCS',
            'sIPS',
            'sPCS',
            'VO']
