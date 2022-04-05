"""
-----------------------------------------------------------------------------------------
fmriprep_sbatch.py
-----------------------------------------------------------------------------------------
Goal of the script:
Run fMRIprep on mesocentre using job mode
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: main project directory
sys.argv[2]: project name (correspond to directory)
sys.argv[3]: subject (e.g. sub-001)
sys.argv[4]: server nb of hour to request (e.g 10)
sys.argv[5]: anat only (1) or not (0)
sys.argv[6]: use of aroma (1) or not (0)
sys.argv[7]: use Use fieldmap-free distortion correction
sys.argv[8]: skip BIDS validation (1) or not (0)
-----------------------------------------------------------------------------------------
Output(s):
preprocessed files
-----------------------------------------------------------------------------------------
To run:
1. cd to function
>> cd /home/mszinte/projects/gaze_prf/analysis_codes/
2. run python command
python preproc/fmriprep_sbatch.py [main directory] [project name] [subject num] [hour proc.] 
                                  [anat only] [aroma] [fmapfree] [skip bids validation]
-----------------------------------------------------------------------------------------
Exemple:
python preproc/fmriprep_sbatch.py /scratch/mszinte/data gaze_prf sub-001 20 1 0 0 0
python preproc/fmriprep_sbatch.py /scratch/mszinte/data gaze_prf sub-001 20 0 0 0 0
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
hour_proc = int(sys.argv[4])
anat = int(sys.argv[5])
aroma = int(sys.argv[6])
fmapfree = int(sys.argv[7])
skip_bids_val = int(sys.argv[8])

# Define cluster/server specific parameters
cluster_name  = 'skylake'
proj_name = 'b161'
singularity_dir = '/scratch/mszinte/softwares/fmriprep-20.1.1.simg'
nb_procs = 32
memory_val = 48
log_dir = opj(main_dir,project_dir,'derivatives','deriv_data','fmriprep','log_outputs')

# special input
anat_only, use_aroma, use_fmapfree, anat_only_end, use_skip_bids_val = '','','','',''
if anat == 1:
	anat_only = ' --anat-only'
	anat_only_end = '_anat'
	nb_procs = 8
if aroma == 1:
	use_aroma = ' --use-aroma'
if fmapfree == 1:
	use_fmapfree= ' --use-syn-sdc'
if skip_bids_val == 1:
	use_skip_bids_val = ' --skip_bids_validation'

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
#SBATCH -e {log_dir}/{subject}_fmriprep{anat_only_end}_%N_%j_%a.err
#SBATCH -o {log_dir}/{subject}_fmriprep{anat_only_end}_%N_%j_%a.out
#SBATCH -J {subject}_fmriprep{anat_only_end}
#SBATCH --mail-type=BEGIN,END\n\n""".format(proj_name = proj_name, nb_procs = nb_procs, hour_proc = hour_proc, subject = subject,
											anat_only_end = anat_only_end, memory_val = memory_val, log_dir = log_dir)

# define singularity cmd
singularity_cmd = "singularity run --cleanenv -B {main_dir}:/work_dir {simg} --fs-license-file /work_dir/freesurfer/license.txt /work_dir/{project_dir}/ /work_dir/{project_dir}/derivatives/deriv_data/fmriprep/ participant --participant-label {sub_num} -w /work_dir/{project_dir}/temp_data/ --bold2t1w-dof 12 --output-spaces T1w MNI152NLin2009cAsym --cifti-output --low-mem --mem-mb 32000 --nthreads {nb_procs:.0f}{anat_only}{use_aroma}{use_fmapfree}{use_skip_bids_val}".format(
									main_dir = main_dir,
									project_dir = project_dir,
									simg = singularity_dir,
									sub_num = sub_num,
									nb_procs = nb_procs,
									anat_only = anat_only,
									use_aroma = use_aroma,
									use_fmapfree = use_fmapfree,
									use_skip_bids_val = use_skip_bids_val)

# create sh folder and file
sh_dir = "{main_dir}/{project_dir}/derivatives/deriv_data/fmriprep/jobs/sub-{sub_num}_fmriprep{anat_only_end}.sh".format(main_dir = main_dir, sub_num = sub_num,project_dir = project_dir,anat_only_end = anat_only_end)

try:
	os.makedirs(opj(main_dir,project_dir,'derivatives','deriv_data','fmriprep','jobs'))
	os.makedirs(opj(main_dir,project_dir,'derivatives','deriv_data','fmriprep','log_outputs'))
except:
	pass
of = open(sh_dir, 'w')
of.write("{slurm_cmd}{singularity_cmd}".format(slurm_cmd = slurm_cmd,singularity_cmd = singularity_cmd))
of.close()

# Submit jobs
print("Submitting {sh_dir} to queue".format(sh_dir = sh_dir))
os.chdir(log_dir)
os.system("sbatch {sh_dir}".format(sh_dir = sh_dir))