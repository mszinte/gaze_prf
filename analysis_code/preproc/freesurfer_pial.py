"""
-----------------------------------------------------------------------------------------
freesurfer_pial.py
-----------------------------------------------------------------------------------------
Goal of the script:
Run freesurfer with new brainmask manually edited
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: main project directory
sys.argv[2]: project name (correspond to directory)
sys.argv[3]: subject (e.g. sub-01)
sys.argv[4]: server nb of hour to request (e.g 10)
-----------------------------------------------------------------------------------------
Output(s):
new freesurfer segmentation files
-----------------------------------------------------------------------------------------
To run:
1. cd to function
>> cd /home/mszinte/projects/gaze_prf/analysis_codes/
2. run python command
python preproc/freesurfer_pial.py [main directory] [project name] [subject] [hour proc.] 
-----------------------------------------------------------------------------------------
Exemple:
python preproc/freesurfer_pial.py /scratch/mszinte/data/ gaze_prf sub-001 20
-----------------------------------------------------------------------------------------
Written by Martin Szinte (martin.szinte@gmail.com)
-----------------------------------------------------------------------------------------
"""

# imports modules
import os
import pdb
import sys
import json
opj = os.path.join
deb = pdb.set_trace

# inputs
main_dir = sys.argv[1]
project_dir = sys.argv[2]
subject = sys.argv[3]
hour_proc = int(sys.argv[4])

# Define cluster/server specific parameters
cluster_name = 'skylake'
nb_procs = 8
memory_val = 48
proj_name = 'b161'
log_dir = opj(main_dir,project_dir,'derivatives','deriv_data','freesurfer_pial','log_outputs')

# define SLURM cmd
slurm_cmd = """\
#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH -p skylake
#SBATCH --mail-user=martin.szinte@univ-amu.fr
#SBATCH -A {proj_name}
#SBATCH --nodes=1
#SBATCH --mem={memory_val}gb
#SBATCH --cpus-per-task={nb_procs}
#SBATCH --time={hour_proc}:00:00
#SBATCH -e {log_dir}/{subject}_freesurfer-pial_%N_%j_%a.err
#SBATCH -o {log_dir}/{subject}_freesurfer-pial_%N_%j_%a.out
#SBATCH -J {subject}_freesurfer-pial
#SBATCH --mail-type=BEGIN,END\n\n""".format(nb_procs = nb_procs, hour_proc = hour_proc, subject = subject,
											memory_val = memory_val, log_dir = log_dir, proj_name = proj_name)

# define subject directory
fs_dir = "{main_dir}{project_dir}/derivatives/deriv_data/fmriprep/freesurfer/".format(main_dir = main_dir, project_dir = project_dir)
fs_licence = '/scratch/mszinte/freesurfer/license.txt'

freesurfer_cmd = """\
export SUBJECTS_DIR={fs_dir}\n\
export FS_LICENSE={fs_licence}\n\
recon-all -autorecon-pial -subjid {subject}""".format(
	fs_dir = fs_dir, fs_licence = fs_licence, subject = subject)

# create sh folder and file
sh_dir = "{main_dir}/{project_dir}/derivatives/deriv_data/freesurfer_pial/jobs/{subject}_freesurfer-pial.sh".format(main_dir = main_dir, subject = subject,project_dir = project_dir)

try:
	os.makedirs(opj(main_dir,project_dir,'derivatives','deriv_data','freesurfer_pial','jobs'))
	os.makedirs(opj(main_dir,project_dir,'derivatives','deriv_data','freesurfer_pial','log_outputs'))
except:
	pass

of = open(sh_dir, 'w')
of.write("{slurm_cmd}{freesurfer_cmd}".format(slurm_cmd = slurm_cmd,freesurfer_cmd = freesurfer_cmd))
of.close()

# Submit jobs
print("Submitting {sh_dir} to queue".format(sh_dir = sh_dir))
os.chdir(log_dir)
os.system("sbatch {sh_dir}".format(sh_dir = sh_dir))