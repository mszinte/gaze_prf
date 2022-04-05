"""
-----------------------------------------------------------------------------------------
pycortex_import.py
-----------------------------------------------------------------------------------------
Goal of the script:
Import subject in pycortex
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: subject name
-----------------------------------------------------------------------------------------
Output(s):
None
-----------------------------------------------------------------------------------------
To run:
cd /home/mszinte/projects/gaze_prd/analysis_codes/
python preproc/pycortex_import.py sub-001
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
import os
import sys
import json
import glob
import numpy as np
import pdb
import platform
opj = os.path.join
deb = pdb.set_trace

# MRI imports
# -----------
import cortex
from cortex.fmriprep import *
import nibabel as nb

# Functions import
# ----------------
from utils import set_pycortex_config_file

# Get inputs
# ----------
subject = sys.argv[1]

# Define analysis parameters
# --------------------------
with open('settings.json') as f:
    json_s = f.read()
    analysis_info = json.loads(json_s)
base_dir = analysis_info['base_dir']

# Define folder
# -------------
fmriprep_dir = "{base_dir}/derivatives/deriv_data/fmriprep/".format(base_dir = base_dir)
fs_dir = "{base_dir}/derivatives/deriv_data/fmriprep/freesurfer/".format(base_dir = base_dir)
temp_dir = "{base_dir}/derivatives/temp_data/{subject}_rand_ds/".format(base_dir = base_dir, subject = subject)
xfm_name = "identity.fmriprep"
cortex_dir = "{base_dir}/derivatives/pp_data/cortex/db/{subject}".format(base_dir = base_dir, subject = subject)

# Set pycortex db and colormaps
# -----------------------------
set_pycortex_config_file(base_dir)

# Add participant to pycortex db
# ------------------------------
print('import subject in pycortex')
cortex.freesurfer.import_subj(fs_subject = subject, cx_subject = subject, freesurfer_subject_dir = fs_dir, whitematter_surf = 'smoothwm')

# Add participant flat maps
# -------------------------
print('import subject flatmaps')
try: cortex.freesurfer.import_flat(fs_subject = subject, cx_subject = subject, freesurfer_subject_dir = fs_dir, patch = 'full', auto_overwrite=True)
except: pass

# Add transform to pycortex db
# ----------------------------
print('add transform')
file_list = sorted(glob.glob("{base_dir}/derivatives/pp_data/{sub}/func/*.nii.gz".format(base_dir = base_dir, sub = subject)))
ref_file = file_list[0]
transform = cortex.xfm.Transform(np.identity(4), ref_file)
transform.save(subject, xfm_name, 'magnet')

# Add masks to pycortex transform
# -------------------------------
print('create pycortex mask')
xfm_masks = analysis_info['xfm_masks']
ref = nb.load(ref_file)
for xfm_mask in xfm_masks:
    
    mask = cortex.get_cortical_mask(subject = subject, xfmname = xfm_name, type = xfm_mask)
    mask_img = nb.Nifti1Image(dataobj=mask.transpose((2,1,0)), affine=ref.affine, header=ref.header)
    mask_file = "{cortex_dir}/transforms/{xfm_name}/mask_{xfm_mask}.nii.gz".format(
                            cortex_dir = cortex_dir, xfm_name = xfm_name, xfm_mask = xfm_mask)
    mask_img.to_filename(mask_file)

# Create participant pycortex overlays
# ------------------------------------
print('create subject pycortex overlays to check')
voxel_vol = cortex.Volume(np.random.randn(mask.shape[0], mask.shape[1], mask.shape[2]), subject = subject, xfmname = xfm_name)
ds = cortex.Dataset(rand=voxel_vol)
cortex.webgl.make_static(outpath = temp_dir, data = ds)
