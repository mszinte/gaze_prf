"""
-----------------------------------------------------------------------------------------
flatten_sbatch.py
-----------------------------------------------------------------------------------------
Goal of the script:
Run mris_flatten on mesocentre using job mode
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: main project directory
sys.argv[2]: project name (correspond to directory)
sys.argv[3]: subject (e.g. sub-001)
sys.argv[4]: hemisphere (r or l)
sys.argv[5]: server nb of hour to request (e.g 10)
-----------------------------------------------------------------------------------------
Output(s):
preprocessed files
-----------------------------------------------------------------------------------------
To run:
1. cd to function
>> cd /home/mszinte/projects/gaze_prf/analysis_codes/
2. run python command
python preproc/flatten_sbatch.py [main directory] [project name] [subject num] 
                                 [hemisphere] [hour proc.]
-----------------------------------------------------------------------------------------
Exemple:
python preproc/flatten_sbatch.py /scratch/mszinte/data gaze_prf sub-001 r 20
-----------------------------------------------------------------------------------------
Written by Martin Szinte (martin.szinte@gmail.com)
-----------------------------------------------------------------------------------------
"""

# imports modules
import sys
import os
import time
import pdb
import json
opj = os.path.join
deb = pdb.set_trace

# inputs
main_dir = sys.argv[1]
project_dir = sys.argv[2]
subject = sys.argv[3]
sub_num = subject[-3:]
hemi = sys.argv[4]
hour_proc = int(sys.argv[5])

# Define cluster/server specific parameters
cluster_name  = 'skylake'
proj_name = 'b161'
nb_procs = 32
memory_val = 48
log_dir = opj(main_dir,project_dir,'derivatives','deriv_data','flatten','log_outputs')
freesurfer_dir = "{main_dir}/{project_dir}/derivatives/deriv_data/fmriprep/freesurfer/".format(main_dir = main_dir,project_dir = project_dir)

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
#SBATCH -e {log_dir}/{subject}_{hemi}h_flatten_%N_%j_%a.err
#SBATCH -o {log_dir}/{subject}_{hemi}h_flatten_%N_%j_%a.out
#SBATCH -J {subject}_{hemi}h_flatten
#SBATCH --mail-type=BEGIN,END
export SUBJECTS_DIR='{freesurfer_dir}'
cd '{freesurfer_dir}{subject}/surf/'\n\n""".format(	proj_name = proj_name, nb_procs = nb_procs, hour_proc = hour_proc, hemi = hemi,
													subject = subject, memory_val = memory_val, log_dir = log_dir, freesurfer_dir = freesurfer_dir)

# define flatten cmd
flatten_cmd = "mris_flatten {hemi}h.full.patch.3d {hemi}h.full.flat.patch.3d".format(hemi = hemi)

# create sh folder and file
sh_dir = "{main_dir}/{project_dir}/derivatives/deriv_data/flatten/jobs/{subject}_{hemi}h_flatten.sh".format(main_dir = main_dir, subject = subject,project_dir = project_dir,hemi = hemi)

try:
	os.makedirs(opj(main_dir,project_dir,'derivatives','deriv_data','flatten','jobs'))
	os.makedirs(opj(main_dir,project_dir,'derivatives','deriv_data','flatten','log_outputs'))
except:
	pass

of = open(sh_dir, 'w')
of.write("{slurm_cmd}{flatten_cmd}".format(slurm_cmd = slurm_cmd,flatten_cmd = flatten_cmd))
of.close()

# Submit jobs
print("Submitting {sh_dir} to queue".format(sh_dir = sh_dir))
os.chdir(log_dir)
os.system("sbatch {sh_dir}".format(sh_dir = sh_dir))
