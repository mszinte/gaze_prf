"""
-----------------------------------------------------------------------------------------
pial_edits.py
-----------------------------------------------------------------------------------------
Goal of the script:
Launch freeview to edit manually the segmentations
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: subject
sys.argv[2]: explanations ('manual_edit')
-----------------------------------------------------------------------------------------
Output(s):
better segmentation files to launch in recon
-----------------------------------------------------------------------------------------
To run localy:
cd ~/disks/meso_H/projects/gaze_prf/analysis_codes/
python preproc/pial_edits.py sub-001 manual_edit
-----------------------------------------------------------------------------------------
Written by Martin Szinte (martin.szinte@gmail.com)
-----------------------------------------------------------------------------------------
"""

# imports modules
import subprocess as sb
import os
import ipdb
import sys
import json
opj = os.path.join
deb = ipdb.set_trace

# inputs
subject = sys.argv[1]
expl = sys.argv[2]

# Define analysis parameters
with open('settings.json') as f:
    json_s = f.read()
    analysis_info = json.loads(json_s)

# define directory
base_dir = analysis_info['base_dir_local']
fs_dir = "{base_dir}/derivatives/deriv_data/fmriprep/freesurfer/{subject}".format(base_dir = base_dir, subject = subject)
edit_dir = "{fs_dir}/edit/{expl}".format(fs_dir = fs_dir, expl = expl)
sh_dir = "{edit_dir}/{subject}_{expl}.sh".format(edit_dir = edit_dir, subject = subject, expl = expl)
try: os.makedirs(edit_dir)
except: pass

# list commands
t1w_cmd = '-v {t1mgz}:grayscale=0,100'.format(t1mgz = '{fs_dir}/mri/T1.mgz'.format(fs_dir = fs_dir))
t2w_cmd = '-v {t2mgz}:grayscale=0,300'.format(t2mgz = '{fs_dir}/mri/T2.mgz'.format(fs_dir = fs_dir))
brainmask_cmd = '-v {brainmask}:grayscale=0,50:opacity=0.4'.format(brainmask = '{fs_dir}/mri/brainmask.mgz'.format(fs_dir = fs_dir))
volumes_cmd = '-f {fs_dir}/surf/lh.white:color=red:edgecolor=red \
-f {fs_dir}/surf/rh.white:color=red:edgecolor=red \
-f {fs_dir}/surf/lh.pial:color=yellow:edgecolor=yellow \
-f {fs_dir}/surf/rh.pial:color=yellow:edgecolor=yellow '.format(fs_dir = fs_dir)

freeview_cmd = '{t2w_cmd} {t1w_cmd} {brainmask_cmd} {volumes_cmd}'.format(
                            t1w_cmd = t1w_cmd,
                            t2w_cmd = t2w_cmd,
                            brainmask_cmd = brainmask_cmd,
                            volumes_cmd = volumes_cmd)

of = open(sh_dir, 'w')
of.write(freeview_cmd)
of.close()

# run freeview cmd
sb.call('freeview -cmd {sh_dir}'.format(sh_dir = sh_dir),shell=True)

# in freeview 
# 1. click on edit button 
# 2. edit the brainmask focussing on occupital lobe and limit with cerebelum
# 3. save the brainmask as
# 4. run freesurfer_pial.py again

