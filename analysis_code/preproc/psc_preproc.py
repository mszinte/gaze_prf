import argparse
import nilearn

from gaze_prf.utils.data import get_roi_mask
from nilearn.input_data import NiftiMasker

from nilearn.signal import clean
from nilearn.image import load_img, clean_img
import os.path as op
import os

def main(subject, session, task, run, bids_folder):


    if type(subject) is int:
        subject = f'{subject:03d}'
    
    if type(session) is int:
        session = f'{session:02d}'

    target_dir = op.join(bids_folder, 'derivatives', 'filtered_data',
                         f'sub-{subject}', f'ses-{session}', 'func')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    func_img = op.join(bids_folder,
                       'deriv_data', 'fmriprep', 'fmriprep',
                       f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-T1w_desc-preproc_bold.nii.gz')

    cleaned_img = clean_img(func_img, standardize='psc', clean_filter='cosine', t_r=1.31, highpass=0.01)
    cleaned_img.to_filename(op.join(target_dir, f'sub-{subject}_ses-{session}_task-{task}_run-{run}_desc-cleaned_bold.nii.gz'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Clean and filter BOLD data')
    parser.add_argument('subject', type=int, help='subject number')
    parser.add_argument('session', type=int, help='session number')
    parser.add_argument('task', type=str, help='task name')
    parser.add_argument('run', type=int, help='run number')
    parser.add_argument('--bids_folder', type=str, help='BIDS folder', default='/tank/shared/2021/visual/pRFgazeMod/')
    args = parser.parse_args()

    main(args.subject, args.session, args.task, args.run, args.bids_folder)