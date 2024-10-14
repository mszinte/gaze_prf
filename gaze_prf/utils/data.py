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
             bids_folder='/tank/shared/2021/visual/pRFgazeMod',
             masker=None):

    fullscreen = (gaze is None)

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
        fn = op.join(bids_folder, f'pp_data/sub-{subject}/func/fmriprep_dct_pca',
                     f'sub-{subject}_ses-{session}_task-{label}_run-{run}_fmriprep_dct_pca.nii.gz')
        keys.append((session, gaze, task, run))
        d = pd.DataFrame(masker.fit_transform(fn))
        d.index.name, d.columns.name = 'time', 'voxel'
        data.append(d)

    data = pd.concat(data, keys=keys, names=['session', 'gaze', 'task', 'run'])

    return data


def get_prf_parameters(subject, session=None, gaze=None, task=None, run=None,
                       masker=None,
                       roi=None,
                       filter=False,
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

            pars = pd.concat(fs_pars, axis=1)
        else:
            pars = pd.read_csv(op.join(bids_folder, 'derivatives', 'prf_fits.mean', f'sub-{subject}', 'func',
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

            pars = pd.concat(pars, axis=1)

        else:
            pars = pd.read_csv(op.join(bids_folder, 'derivatives', 'prf_fits', f'sub-{subject}', 'func',
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

            pars = pd.concat(pars, axis=1)

        else:
            get_data = pd.read_csv(op.join(bids_folder, 'derivatives', 'prf_fits', f'sub-{subject}', 'func',
                                       f'sub-{subject}_task-{task}Gaze{gaze}_roi-{roi}_desc-gaussprf.optim.parameters.tsv'), sep='\t', index_col=0)

    else:
        raise NotImplementedError()

    if filter:
        return filter_pars(pars)
    else:
        return pars


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

    prf_pars = get_prf_parameters(subject=subject, session=session, gaze=gaze, task=task, run=run, masker=masker,
                                  roi=roi, bids_folder=bids_folder)

    model = GaussianPRF2DWithHRF(
        grid_coordinates, paradigm, hrf_model=hrf_model,
        parameters=prf_pars)

    return model


def get_brain_mask(subject, bids_folder='/tank/shared/2021/visual/pRFgazeMod/',
                   resample_to_func=True):

    if resample_to_func:
        mask = image.load_img(op.join(bids_folder, f'deriv_data/fmriprep/fmriprep/',
                                    f'sub-{subject}', 'anat', f'sub-{subject}_desc-brain_space-func_mask.nii.gz'))
        print('yo')
    else:
        mask = image.load_img(op.join(bids_folder, f'deriv_data/fmriprep/fmriprep/',
                                    f'sub-{subject}', 'anat', f'sub-{subject}_desc-brain_mask.nii.gz'))
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
    elif hemi == 'both':
        mask_l = image.load_img(op.join(folder, f'{mask}_cortical_L.nii.gz'))
        mask_r = image.load_img(op.join(folder, f'{mask}_cortical_R.nii.gz'))
        return image.math_img('mask_l+mask_r', mask_l=mask_l, mask_r=mask_r)
    else:
        raise ValueError(f'{hemi} is not a valid hemisphere, use L/R/both')


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


def filter_pars(pars):
    # Threshold values
    amp_th = 0
    r2_th = [0, 1]
    ecc_th = [0.05, 15]
    sd_th = [0.05, 15]

    pars.loc[pars.amplitude < amp_th] = np.nan
    pars.loc[(pars.r2 < r2_th[0]) | (pars.r2 > r2_th[1])] = np.nan
    pars.loc[(pars.ecc < ecc_th[0]) | (pars.ecc > ecc_th[1])] = np.nan
    pars.loc[(pars.sd < sd_th[0]) | (pars.sd > sd_th[1])] = np.nan

    return pars


def get_fixation_screen(bids_folder='/tank/shared/2021/visual/pRFgazeMod', resize_factor=1., max_intensity=255.):
    x, y = grid_coordinates = get_grid_coordinates(
        bids_folder=bids_folder, resize_factor=1., flattened=False)
    fix_screen = (np.sqrt(x**2 + y**2) < .4) * max_intensity
    fix_screen = ndimage.zoom(fix_screen, 1./resize_factor)

    return fix_screen


def get_dm_parameters(bids_folder='/tank/shared/2021/visual/pRFgazeMod', include_xy=True,
                      task='FS'):
    dm = get_dm(bids_folder=bids_folder)
    width_degrees = np.arctan(69/2. / 220) / (2*np.pi) * 360 * 2
    height_degrees = dm.shape[2] / dm.shape[1] * width_degrees

    # Things were mirrored
    # width_degrees, height_degrees = height_degrees, width_degrees

    if task == 'FS':
        radius_wide = np.linspace(-width_degrees / 2.,
                                  width_degrees / 2, 32, endpoint=True)
        radius_height = np.linspace(-height_degrees / 2.,
                                    height_degrees / 2, 18, endpoint=True)[::-1]

        parameters = pd.DataFrame(np.ones((150, 3)) * np.nan, columns=[
            'angle', 'radius', 'width'])
    elif task == 'aperture':
        radius_wide = np.linspace(-width_degrees / 4.,
                                  width_degrees / 4, 18, endpoint=True)
        parameters = pd.DataFrame(np.ones((122, 3)) * np.nan, columns=[
            'angle', 'radius', 'width'])

        parameters['height'] = pd.read_csv(op.join(bids_folder, 'pp_data', 'visual_dm', 'bar_height.tsv'), sep='\t',
                                           index_col=0)

    parameters.index.name = 'time'
    parameters.columns.name = 'parameter'

    bar_width = 19 / 240 * width_degrees

    if task == 'FS':
        parameters.loc[10:41, 'width'] = bar_width
        parameters.loc[10:41, 'radius'] = radius_wide[::-1]
        parameters.loc[10:41, 'angle'] = 0.0

        parameters.loc[52:69, 'width'] = bar_width
        parameters.loc[52:69, 'radius'] = radius_height
        parameters.loc[52:69, 'angle'] = .5 * np.pi

        parameters.loc[80:111, 'width'] = bar_width
        parameters.loc[80:111, 'radius'] = radius_wide
        parameters.loc[80:111, 'angle'] = 0.0

        parameters.loc[122:139, 'width'] = bar_width
        parameters.loc[122:139, 'radius'] = radius_height[::-1]
        parameters.loc[122:139, 'angle'] = .5 * np.pi
    elif task == 'aperture':
        parameters.loc[10:27, 'width'] = bar_width
        parameters.loc[10:27, 'radius'] = radius_wide[::-1]
        parameters.loc[10:27, 'angle'] = 0.0

        parameters.loc[38:55, 'width'] = bar_width
        parameters.loc[38:55, 'radius'] = radius_wide
        parameters.loc[38:55, 'angle'] = 0.0

        parameters.loc[66:83, 'width'] = bar_width
        parameters.loc[66:83, 'radius'] = radius_wide[::-1]
        parameters.loc[66:83, 'angle'] = 0.0

        parameters.loc[94:111, 'width'] = bar_width
        parameters.loc[94:111, 'radius'] = radius_wide
        parameters.loc[94:111, 'angle'] = 0.0

    if include_xy:
        parameters['x'] = np.cos(parameters['angle']) * parameters['radius']

        if task == 'FS':
            parameters['y'] = np.sin(
                parameters['angle']) * parameters['radius']
            parameters.at[parameters['y'].abs() < 1e-3, 'y'] = 0.0

        parameters['ecc'] = parameters['radius'].abs()
        parameters.at[parameters['x'].abs() < 1e-3, 'x'] = 0.0

    return parameters


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


