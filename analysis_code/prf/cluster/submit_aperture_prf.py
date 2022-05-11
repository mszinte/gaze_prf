#!/usr/bin/env python

import os
import os.path as op
from datetime import datetime
from itertools import product
import numpy as np
import pandas as pd

job_directory = "%s/.job" %os.getcwd()

if not op.exists(job_directory):
    os.makedirs(job_directory)

subjects = ['%03d' % i for i in range(1, 9)]
tasks = ['AttendFix', 'AttendStim']
gazes = ['Left', 'Center', 'Right']
starting_models = ['retinotopic', 'spatiotopic']

for ix, (subject, task, gaze, starting_model) in enumerate(product(subjects, tasks, gazes, starting_models)):
    print(f'*** RUNNING {subject}, {task}, {gaze}, {starting_model} ')

    job_file = os.path.join(job_directory, f"{ix}.job")
    
    now = datetime.now()
    time_str = now.strftime("%Y.%m.%d_%H.%M.%S")

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/sh\n")
        id = f'{subject}.{task}.{gaze}.{starting_model}.{ix}'
        fh.writelines(f"#SBATCH --job-name=fit_aperture_prf.{id}.job\n")
        fh.writelines(f"#SBATCH --output={os.environ['HOME']}/.out/fit_aperture_prf.{id}.out\n")
        fh.writelines(f"#SBATCH --error={os.environ['HOME']}/.out/fit_aperture_prfs.{id}.err\n")
        fh.writelines("#SBATCH --partition=gpu_shared --gpus-per-node=1\n")
        fh.writelines("#SBATCH --time=90:00\n")
        fh.writelines('source $HOME/.bash_profile\n')
        fh.writelines("conda activate tf-gpu\n")
        cmd = f"python $HOME/git/gaze_prf/analysis_code/prf/refine_aperture_conditions.py {subject} {starting_model} --task {task} --gaze {gaze} --bids_folder $HOME/data/ds-prf\n"
        fh.writelines(cmd)
        print(cmd)

    os.system("sbatch %s" %job_file)
