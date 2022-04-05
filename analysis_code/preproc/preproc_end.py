"""
-----------------------------------------------------------------------------------------
pre_fit_end.py
-----------------------------------------------------------------------------------------
Goal of the script:
Arrange data
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: subject name
-----------------------------------------------------------------------------------------
Output(s):
arranged data in pp_data
-----------------------------------------------------------------------------------------
To run:
1. cd to function
>> cd /home/mszinte/projects/gaze_prf/analysis_codes/
2. python preproc/preproc_end.py sub-001
-----------------------------------------------------------------------------------------
Written by Martin Szinte (martin.szinte@gmail.com)
-----------------------------------------------------------------------------------------
"""

# Stop warnings
# -------------
import warnings
warnings.filterwarnings("ignore")

# General imports
# ---------------
import json
import sys
import os
import glob
import pdb
import platform
import numpy as np
opj = os.path.join
deb = pdb.set_trace

sub_name = sys.argv[1]

# MRI analysis imports
# --------------------
import nibabel as nb
from scipy.signal import savgol_filter

with open('settings.json') as f:
    json_s = f.read()
    analysis_info = json.loads(json_s)
trans_cmd = 'rsync -avuz --progress'

# Define cluster/server specific parameters
# -----------------------------------------
base_dir = analysis_info['base_dir'] 

# Copy files in pp_data folder
# ----------------------------
dest_folder1 = "{base_dir}/derivatives/pp_data/{sub}/".format(base_dir = base_dir, sub = sub_name)
try: os.makedirs(dest_folder1)
except: pass

orig_folder = "{base_dir}/derivatives/deriv_data/pybest/{sub}".format(base_dir = base_dir, sub = sub_name)

for session in analysis_info['session']:

    for attend_cond in analysis_info['attend_cond']:
        for gaze_cond in analysis_info['gaze_cond']:
            if gaze_cond == 'GazeCenterFS':
                runs = analysis_info['runs']
                run_txt = ['_run-1','_run-2']

            else:
                runs = [analysis_info['runs'][0]]
                run_txt = ['']

            for run,run_txt in zip(runs,run_txt):
                # dct func
                orig_file1 = "{orig_fold}/{session}/preproc/{sub}_{session}_task-{attend_cond}{gaze_cond}_space-T1w{run_txt}_desc-preproc_bold.nii.gz".format(
                                        orig_fold = orig_folder, session = session, run = run, run_txt = run_txt,
                                        sub = sub_name, attend_cond = attend_cond, gaze_cond = gaze_cond)
                dest_file1 = "{dest_fold}/{sub}_{session}_task-{attend_cond}{gaze_cond}_{run}_fmriprep_dct.nii.gz".format(
                                        dest_fold = dest_folder1, session = session, run = run,
                                        sub = sub_name, attend_cond = attend_cond, gaze_cond = gaze_cond)

                os.system("{cmd} {orig} {dest}".format(cmd = trans_cmd, orig = orig_file1, dest = dest_file1))


