"""
-----------------------------------------------------------------------------------------
deface_sbatch.py
-----------------------------------------------------------------------------------------
Goal of the script:
Deface T1w and T2w images
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: main project directory
sys.argv[2]: project name (correspond to directory)
sys.argv[3]: subject (e.g. sub-001)
sys.argv[4]: overwrite images (0 = no, 1 = yes)
-----------------------------------------------------------------------------------------
Output(s):
Defaced 
-----------------------------------------------------------------------------------------
To run:
1. cd to function
>> cd /home/mszinte/projects/gaze_prf/mri_analysis/
2. run python command
python preproc/deface_sbatch.py [main directory] [project name] [subject num] [overwrite] 
-----------------------------------------------------------------------------------------
Exemple:
python preproc/deface_sbatch.py /scratch/mszinte/data gaze_prf sub-001 1
-----------------------------------------------------------------------------------------
Written by Martin Szinte (martin.szinte@gmail.com)
-----------------------------------------------------------------------------------------
"""

# imports modules
import sys
import os
import pdb
from bids import BIDSLayout
deb = pdb.set_trace

# inputs
main_dir = sys.argv[1]
project_dir = sys.argv[2]
subject = sys.argv[3]
ovewrite_in = int(sys.argv[4])
hour_proc = 4
nb_procs = 8
log_dir = "{}/{}/derivatives/pp_data/logs".format(main_dir, project_dir, subject)
try: os.makedirs(log_dir)
except: pass

# define SLURM cmd
slurm_cmd = """\
#!/bin/bash
#SBATCH -p skylake
#SBATCH -A b161
#SBATCH --nodes=1
#SBATCH --cpus-per-task={nb_procs}
#SBATCH --time={hour_proc}:00:00
#SBATCH -e {log_dir}/{subject}_deface_%N_%j_%a.err
#SBATCH -o {log_dir}/{subject}_deface_%N_%j_%a.out
#SBATCH -J {subject}_deface\n\n""".format(nb_procs=nb_procs, hour_proc=hour_proc, subject=subject, log_dir=log_dir)

# get files
layout = BIDSLayout("{}/{}/".format(main_dir,project_dir,subject), 'synthetic')
t1w_filenames = layout.get(subject='001',suffix='T1w',extension='nii.gz',return_type='filename')
t2w_filenames = layout.get(subject='001',suffix='T2w',extension='nii.gz',return_type='filename')
deface_filenames = t1w_filenames + t2w_filenames

# sh folder & file
sh_folder = "{}/{}/derivatives/pp_data/jobs".format(main_dir, project_dir, subject)
try: os.makedirs(sh_folder)
except: pass
sh_file = "{}/{}_deface.sh".format(sh_folder,subject)

of = open(sh_file, 'w')
of.write(slurm_cmd)
for deface_filename in deface_filenames:
    if ovewrite_in == 1: deface_cmd = "pydeface {fn} --outfile {fn} --force --verbose\n".format(fn = deface_filename)
    else: deface_cmd = "pydeface {fn} --verbose/n".format(fn = deface_filename)
    
    of.write(deface_cmd)    
of.close()

# Submit jobs
print("Submitting {} to queue".format(sh_file))
os.chdir(log_dir)
os.system("sbatch {}".format(sh_file))