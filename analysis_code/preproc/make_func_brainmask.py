import argparse
from gaze_prf.utils.data import resample_mask_to_functional_space
from nilearn import image
import os.path as op

def main(subject, bids_folder='/tank/shared/2021/visual/pRFGazemod'):
    mask = image.load_img(op.join(bids_folder, f'deriv_data/fmriprep/fmriprep/',
                                f'sub-{subject}', 'anat', f'sub-{subject}_desc-brain_mask.nii.gz'))

    mask_resample = resample_mask_to_functional_space(subject, mask, bids_folder)

    mask_resample.to_filename(op.join(bids_folder, f'deriv_data/fmriprep/fmriprep/', f'sub-{subject}', 'anat', f'sub-{subject}_desc-brain_space-func_mask.nii.gz'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument(
        '--bids_folder', default='/tank/shared/2021/visual/pRFgazeMod/')

    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder)





