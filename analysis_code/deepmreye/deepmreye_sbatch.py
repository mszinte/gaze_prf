"""
-----------------------------------------------------------------------------------------
deepmreye_sbatch.py
-----------------------------------------------------------------------------------------
Goal of the script:
Run DeepMReye on mesocentre using job mode
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: main project directory
sys.argv[2]: project name (correspond to directory)
sys.argv[3]: server nb of hour to request (e.g 10)
sys.argv[3]: GPU ? (if not CPU)
sys.argv[5]: email account (e.g. martin.szinte@univ-amu.fr
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
python deepmreye_sbatch.py /scratch/mszinte/data gaze_prf GazeCenterFS 10 martin.szinte@univ-amu.fr 327 b327
python deepmreye_sbatch.py /scratch/mszinte/data gaze_prf GazeCenter 10 martin.szinte@univ-amu.fr 327 b327
python deepmreye_sbatch.py /scratch/mszinte/data gaze_prf GazeLeft 10 martin.szinte@univ-amu.fr 327 b327
python deepmreye_sbatch.py /scratch/mszinte/data gaze_prf GazeRight 10 martin.szinte@univ-amu.fr 327 b327
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
email = sys.argv[5]
group = sys.argv[6]
server_project = sys.argv[7]

# Define cluster/server specific parameters
nb_procs = 8
cluster_name  = 'skylake'
memory_val = 100
log_dir = f"{main_dir}/{project_dir}/derivatives/deepmreye/log_outputs"

# define SLURM cmd
slurm_cmd = """\
#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH -p {cluster_name}
#SBATCH --mail-user={email}
#SBATCH -A {server_project}
#SBATCH --nodes=1
#SBATCH --mem={memory_val}gb
#SBATCH --cpus-per-task={nb_procs}
#SBATCH --time={hour_proc}:00:00
#SBATCH -e {log_dir}/deepmreye_%N_%j_%a.err
#SBATCH -o {log_dir}/deepmreye_%N_%j_%a.out
#SBATCH -J deepmreye_{gaze_task}
#SBATCH --mail-type=BEGIN,END\n\n
""".format(server_project=server_project, nb_procs=nb_procs, hour_proc=hour_proc, gaze_task=gaze_task,
		   memory_val=memory_val, log_dir=log_dir, email=email, cluster_name=cluster_name)

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