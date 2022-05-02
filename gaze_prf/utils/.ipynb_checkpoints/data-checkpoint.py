import itertools
import pandas as pd
import os.path as op
import numpy as np
from scipy import ndimage
from scipy.io import loadmat
from braincoder.models import LinearModelWithBaselineHRF, GaussianPRF2DWithHRF
from braincoder.hrf import CustomHRFModel


def get_dm(sourcedata='/tank/shared/2021/visual/pRFgazeMod/', square=False, resize_factor=1,
           task='GazeCenterFS'):
    dms = loadmat(op.join(sourcedata, 'pp_data', 'visual_dm',
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

    return dms


def get_grid_coordinates(sourcedata='/tank/shared/2021/visual/pRFgazeMod', flattened=True, resize_factor=1):

    dm = get_dm(sourcedata=sourcedata)
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


def get_prfpy_hrf(sourcedata='/tank/shared/2021/visual/pRFgazeMod'):

    return np.loadtxt(op.join(sourcedata, 'derivatives', 'hrf.txt'))

def get_data(subject='001', session='01', task='AttendFixGazeCenterFS', run=1, roi='V1',
             sourcedata='/tank/shared/2021/visual/pRFgazeMod',
             mask=None):

    data = np.load(op.join(sourcedata, 'pp_data', f'sub-{subject}', 'masked', roi, 'timecourses',
                           f'sub-{subject}_ses-{session}_task-{task}_run-{run}_fmriprep_dct_{roi}.npy'))

    data = pd.DataFrame(data)

    if mask is not None:
        data = data.loc[mask, :]

    return data.T
