"""
-----------------------------------------------------------------------------------------
deepmreye_gpu_sbatch.py
-----------------------------------------------------------------------------------------
Goal of the script:
Run DeepMReye on mesocentre using job mode
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: main project directory
sys.argv[2]: project name (correspond to directory)
sys.argv[3]: gaze task
sys.argv[3]: server nb of hour to request (e.g 10)
sys.argv[5]: data group (e.g. 327)
sys.argv[6]: project name (e.g. b327)
-----------------------------------------------------------------------------------------
Output(s):
eye position
-----------------------------------------------------------------------------------------
To run:
1. cd to function
>> cd ~/projects/[PROJECT_DIR]/deepmreye/
2. run python command
python deepmreye_sbatch [main directory] [project name] [gaze_task] [hour proc.] 
						[email account] [group] [server_project]
-----------------------------------------------------------------------------------------
Exemple:
cd ~/projects/gaze_prf/analysis_code/deepmreye/
python deepmreye_sbatch.py /scratch/mszinte/data gaze_prf GazeCenterFS 1 327 b327
python deepmreye_sbatch.py /scratch/mszinte/data gaze_prf GazeCenter 1 327 b327
python deepmreye_sbatch.py /scratch/mszinte/data gaze_prf GazeLeft 1 327 b327
python deepmreye_sbatch.py /scratch/mszinte/data gaze_prf GazeRight 1 327 b327
-----------------------------------------------------------------------------------------
Written by Martin Szinte (martin.szinte@gmail.com)
-----------------------------------------------------------------------------------------
"""
# Debug 
import ipdb
deb = ipdb.set_trace

# imports modules
import os
import sys
opj = os.path.join

# inputs
main_dir = sys.argv[1]
project_dir = sys.argv[2]
gaze_task = sys.argv[3]
hour_proc = int(sys.argv[4])
group = sys.argv[5]
server_project = sys.argv[6]

# Define cluster/server specific parameters
nb_procs = 1
cluster_name  = 'volta'
log_dir = f"{main_dir}/{project_dir}/derivatives/deepmreye/log_outputs"

# define SLURM cmd
slurm_cmd = """\
#!/bin/bash
#SBATCH -p {cluster_name}
#SBATCH -A {server_project}
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time={hour_proc}:00:00
#SBATCH -e {log_dir}/deepmreye_gpu_%N_%j_%a.err
#SBATCH -o {log_dir}/deepmreye_gpu_%N_%j_%a.out
#SBATCH -J deepmreye_gpu_{gaze_task}\n\n
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH\n
""".format(server_project=server_project, nb_procs=nb_procs, hour_proc=hour_proc, gaze_task=gaze_task,
		   log_dir=log_dir, email=email, cluster_name=cluster_name)

# Define main cmd
main_cmd = 'python deepmreye_analysis.py {} {} {} {}'.format(main_dir, project_dir, gaze_task, group)

# create sh folder and file
sh_fn = "{}/{}/derivatives/deepmreye/jobs/deepmreye_{}.sh".format(main_dir, project_dir, gaze_task)

os.makedirs("{main_dir}/{project_dir}/derivatives/deepmreye/jobs".format(
                main_dir=main_dir,project_dir=project_dir), exist_ok=True)
os.makedirs("{main_dir}/{project_dir}/derivatives/deepmreye/log_outputs".format(
                main_dir=main_dir,project_dir=project_dir), exist_ok=True)

of = open(sh_fn, 'w')
of.write("{} \n{} ".format(slurm_cmd, main_cmd))
of.close()

# Submit jobs
print("Submitting {sh_fn} to queue".format(sh_fn=sh_fn))
os.system("sbatch {sh_fn}".format(sh_fn=sh_fn))