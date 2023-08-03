"""
-----------------------------------------------------------------------------------------
average_runs.py
-----------------------------------------------------------------------------------------
Goal of the script:
Average the runs
-----------------------------------------------------------------------------------------
Input(s):
-----------------------------------------------------------------------------------------
Output(s):
Average runs nifti
-----------------------------------------------------------------------------------------
To run:
1. cd to function
>> cd /home/mszinte/projects/gaze_prf/analysis_code/
2. python preproc/average_runs.py
-----------------------------------------------------------------------------------------
Written by Martin Szinte (martin.szinte@gmail.com)
-----------------------------------------------------------------------------------------
"""
# Imports
import os
import numpy as np
import warnings
import glob
import nibabel as nb
warnings.filterwarnings('ignore')

# Define parameters
subjects = ['sub-001', 'sub-002', 'sub-003', 'sub-004',
            'sub-005', 'sub-006', 'sub-007', 'sub-008']

# Define folders
base_dir = '/scratch/mszinte/data/gaze_prf'
bids_dir = "{}".format(base_dir)
pp_dir = "{}/derivatives/pp_data".format(base_dir)

for subject in subjects:
    
    func_dir = '{}/{}/func'.format(pp_dir, subject)
    func_avg_dir = '{}/{}/func_avg'.format(pp_dir, subject)
    try: os.makedirs(func_avg_dir)
    except: pass
    
    # FullScreen runs
    fs_runs = sorted(glob.glob('{}/*GazeCenterFS*.nii.gz'.format(func_dir,subject)))
    fs_avg_fn = '{}/{}_task-FullScreen_fmriprep_dct_avg.nii.gz'.format(func_avg_dir,subject)
    
    # FullScreen AttendFix / AttendBar
    fs_af_runs = sorted(glob.glob('{}/*AttendFixGazeCenterFS*.nii.gz'.format(func_dir,subject)))
    fs_af_avg_fn = '{}/{}_task-FullScreenAttendFix_fmriprep_dct_avg.nii.gz'.format(func_avg_dir,subject)
    fs_ab_runs = sorted(glob.glob('{}/*AttendStimGazeCenterFS*.nii.gz'.format(func_dir,subject)))
    fs_ab_avg_fn = '{}/{}_task-FullScreenAttendBar_fmriprep_dct_avg.nii.gz'.format(func_avg_dir,subject)
    
    # GazeCenter AttendFix / AttendBar
    gc_af_runs = sorted(glob.glob('{}/*AttendFixGazeCenter_*.nii.gz'.format(func_dir,subject)))
    gc_af_avg_fn = '{}/{}_task-GazeCenterAttendFix_fmriprep_dct_avg.nii.gz'.format(func_avg_dir,subject)
    gc_ab_runs = sorted(glob.glob('{}/*AttendStimGazeCenter_*.nii.gz'.format(func_dir,subject)))
    gc_ab_avg_fn = '{}/{}_task-GazeCenterAttendBar_fmriprep_dct_avg.nii.gz'.format(func_avg_dir,subject)
    
    # GazeLeft AttendFix / AttendBar
    gl_af_runs = sorted(glob.glob('{}/*AttendFixGazeLeft_*.nii.gz'.format(func_dir,subject)))
    gl_af_avg_fn = '{}/{}_task-GazeLeftAttendFix_fmriprep_dct_avg.nii.gz'.format(func_avg_dir,subject)
    gl_ab_runs = sorted(glob.glob('{}/*AttendStimGazeLeft_*.nii.gz'.format(func_dir,subject)))
    gl_ab_avg_fn = '{}/{}_task-GazeLeftAttendBar_fmriprep_dct_avg.nii.gz'.format(func_avg_dir,subject)
    
    # GazeRight AttendFix / AttendBar
    gr_af_runs = sorted(glob.glob('{}/*AttendFixGazeRight_*.nii.gz'.format(func_dir,subject)))
    gr_af_avg_fn = '{}/{}_task-GazeRightAttendFix_fmriprep_dct_avg.nii.gz'.format(func_avg_dir,subject)
    gr_ab_runs = sorted(glob.glob('{}/*AttendStimGazeRight_*.nii.gz'.format(func_dir,subject)))
    gr_ab_avg_fn = '{}/{}_task-GazeRightAttendBar_fmriprep_dct_avg.nii.gz'.format(func_avg_dir,subject)
    
    task_runs = [fs_runs, fs_af_runs, fs_ab_runs, 
                 gc_af_runs, gc_ab_runs,
                 gl_af_runs, gl_ab_runs,
                 gr_af_runs, gr_ab_runs]
    
    avg_fns = [fs_avg_fn, fs_af_avg_fn, fs_ab_avg_fn,
               gc_af_avg_fn, gc_ab_avg_fn,
               gl_af_avg_fn, gl_ab_avg_fn,
               gr_af_avg_fn, gr_ab_avg_fn,]
    
    for runs,avg_fn in zip(task_runs,avg_fns):    
        for run_num, run in enumerate(runs):
            print('adding: {}'.format(run))
            if run_num == 0: mat_avg = np.zeros(nb.load(run).shape)
            mat_avg += nb.load(run).get_fdata()/len(runs)

        print('saving: {}'.format(avg_fn))
        avg_img = nb.Nifti1Image(dataobj=mat_avg, affine=nb.load(run).affine, header=nb.load(run).header)
        avg_img.to_filename(avg_fn)
