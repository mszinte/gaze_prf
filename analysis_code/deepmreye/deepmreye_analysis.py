"""
-----------------------------------------------------------------------------------------
deepmreye_analysis.py
-----------------------------------------------------------------------------------------
Goal of the script:
Run deepmreye on fmriprep output 
-----------------------------------------------------------------------------------------
Input(s):
-----------------------------------------------------------------------------------------
Output(s):
TSV with gaze position
-----------------------------------------------------------------------------------------
To run:
1. cd to function
>> cd /home/mszinte/projects/gaze_prf/analysis_code/deepmreye
2. python deepmreye_analysis.py [main directory] [project name] [group]
-----------------------------------------------------------------------------------------
Exemple:
cd ~/projects/gaze_prf/analysis_code/deepmreye/
python deepmreye_analysis.py /scratch/mszinte/data gaze_prf 327 
-----------------------------------------------------------------------------------------
Written by Martin Szinte (martin.szinte@gmail.com)
-----------------------------------------------------------------------------------------
"""

# Import modules and add library to path
import ipdb
import sys
import json
import os
import pickle
import glob
import warnings
import numpy as np
import pandas as pd
import plotly.express as px

# DeepMReye imports
from deepmreye import analyse, preprocess, train
from deepmreye.util import data_generator, model_opts 

def adapt_evaluation(evaluation, npz_fn, cor_xy, TR_dur, subTRs_num, TR_start, TR_end):

    # Get X/Y values across subTR
    pred = evaluation[npz_fn]["pred_y"][TR_start:TR_end,:,:]
    pred_conc = np.concatenate(pred, axis=0)
    
    # Correct them for range
    pred_conc_cor = pred_conc * cor_xy
    
    # Create timestamps in seconds
    timestamps = np.arange(pred_conc.shape[0]+1)
    timestamps = timestamps[1:]*(TR_dur/subTRs_num)
    
    # Create dataframe and save
    preds = np.column_stack((timestamps, pred_conc, pred_conc_cor))
    df = pd.DataFrame(preds, columns=['timestamps', 'x_coord', 'y_coord', 'x_coord_cor', 'y_coord_cor'])

    return df

# Define paths to functional data
main_dir = f"{sys.argv[1]}/{sys.argv[2]}/derivatives/deepmreye"
func_dir = f"{main_dir}/func"
model_dir = f"{main_dir}/model/"
model_file = f"{model_dir}datasets_1to5.h5"
pp_dir = f"{main_dir}/pp_data"
mask_dir = f"{main_dir}/mask"
report_dir = f"{main_dir}/report"
pred_dir = f"{main_dir}/pred"

# Make directories
os.makedirs(pp_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)

# Define settings
with open('settings.json') as f:
    json_s = f.read()
    settings = json.loads(json_s)

subjects = settings['subjects']
gaze_tasks = [f"{sys.argv[3]}_"]
subTRs = settings['subTRs']
TR = settings['TR']
num_res_col = settings['num_res_col']
deepmreye_max_xy = settings['deepmreye_max_xy']
exp_max_xy = settings['exp_max_xy']
noise_std = settings['noise_std']
num_gaze_centerfs = settings['num_gaze_centerfs']
num_gaze_center_left_right = settings['num_gaze_center_left_right']
opts = model_opts.get_opts()
opts['epochs'] = settings['epochs']
opts['batch_size'] = settings['batch_size']
opts['steps_per_epoch'] = settings['steps_per_epoch']
opts['validation_steps'] = settings['validation_steps']
opts['lr'] = settings['lr']
opts['lr_decay'] = settings['lr_decay']
opts['rotation_y'] = settings['rotation_y']
opts['rotation_x'] = settings['rotation_x']
opts['rotation_z'] = settings['rotation_z']
opts['shift'] = settings['shift']
opts['zoom'] = settings['zoom']
opts['gaussian_noise'] = settings['gaussian_noise']
opts['mc_dropout'] = settings['mc_dropout']
opts['dropout_rate'] = settings['dropout_rate']
opts['loss_euclidean'] = settings['loss_euclidean']
opts['error_weighting'] = settings['error_weighting']
opts['num_fc'] = settings['num_fc']
opts['load_pretrained'] = model_file
opts["train_test_split"] = settings['train_test_split']

# Define environment cuda
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # stop warning
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Change to os.environ["CUDA_VISIBLE_DEVICES"] = "" if you dont have access to a GPU


# Preload masks to save time within subject loop
(eyemask_small, eyemask_big, dme_template, mask, x_edges, y_edges, z_edges) = preprocess.get_masks()

for subject in subjects:
	print(f"Running {subject}")
	func_sub_dir = f"{func_dir}/{subject}"
	mask_sub_dir = f"{mask_dir}/{subject}"
	func_files = glob.glob(f"{func_sub_dir}/*.nii.gz")

	for func_file in func_files:
		mask_file_orig = f"{func_sub_dir}/mask_{os.path.basename(func_file)[:-7]}.p"
		mask_file_dest = f"{mask_sub_dir}/mask_{os.path.basename(func_file)[:-7]}.p"

		if os.path.exists(mask_file_orig):
			print(f"{mask_file_orig} exists.")
		elif os.path.exists(mask_file_dest):
			print(f"{mask_file_dest} exists.")
		else:
			print(mask_file_orig)
			preprocess.run_participant(fp_func=func_file, 
									   dme_template=dme_template, 
									   eyemask_big=eyemask_big, 
									   eyemask_small=eyemask_small,
									   x_edges=x_edges, y_edges=y_edges, z_edges=z_edges,
									   transforms=['Affine', 'Affine', 'SyNAggro'])

# Move to destination folder
for subject in subjects:
	func_sub_dir = f"{func_dir}/{subject}"
	
	# .p files
	rsync_mask_cmd = f"rsync -avuz {func_sub_dir}/*.p {mask_dir}/{subject}/"
	rm_mask_cmd = f"rm -Rf {func_sub_dir}/*.p"
	os.system(rsync_mask_cmd)
	os.system(rm_mask_cmd)
	
	# .html files
	rsync_report_cmd = f"rsync -avuz --remove-source-files {func_sub_dir}/*.html {report_dir}/{subject}/"
	rm_report_cmd = f"rm -Rf {func_sub_dir}/*.html"
	os.system(rsync_report_cmd)
	os.system(rm_report_cmd)

# Pre-process data
for gaze_task in gaze_tasks: 
    for subject in subjects:    
        subject_data = []
        subject_labels = [] 
        subject_ids = []

        mask_sub_dir = f"{mask_dir}/{subject}"
        pp_task_dir = f"{pp_dir}/{gaze_task[:-1]}"
        os.makedirs(pp_task_dir, exist_ok=True)
        
        mask_files = glob.glob(f"{mask_sub_dir}/*{gaze_task}*.p")
        if gaze_task == 'GazeCenterFS_': label_coord = np.array([0, 0])
        elif gaze_task == 'GazeCenter_': label_coord = np.array([0, 0])
        elif gaze_task == 'GazeLeft_': label_coord = np.array([-4, 0])
        elif gaze_task == 'GazeRight_': label_coord = np.array([+4, 0])

        for mask_file_num, mask_file in enumerate(mask_files):
            mask = pickle.load(open(mask_file, "rb"))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mask = preprocess.normalize_img(mask)

            labels = np.random.normal(loc=label_coord, scale=noise_std, size=(mask.shape[3], subTRs, num_res_col))

            if mask.shape[3] != labels.shape[0]:
                print(f"WARNING --- Skipping Subject {subject} Run {mask_file_num+1} "
                      f"--- Wrong alignment (Mask {mask.shape} - Label {labels.shape}).")
                continue

            ids = ([subject] * labels.shape[0],
                   [gaze_task] * labels.shape[0],
                   [mask_file_num+1] * labels.shape[0])

            subject_data.append(mask)
            subject_labels.append(labels)
            subject_ids.append(ids)

        # Save participant file
        preprocess.save_data(participant=f"{subject}_{gaze_task}label",
                             participant_data=subject_data,
                             participant_labels=subject_labels,
                             participant_ids=subject_ids,
                             processed_data=pp_task_dir,
                             center_labels=False)

    try:
        os.system(f'rm {pp_task_dir}/.DS_Store')
        print('.DS_Store file deleted successfully.')
    except Exception as e:
        print(f'An error occurred: {e}')

# Train and evaluate model
evaluation, scores = dict(), dict()
for gaze_task in gaze_tasks: 
    # define filepath
    pp_task_dir = f"{pp_dir}/{gaze_task[:-1]}"
    dataset_name = f"gaze_prf_{gaze_task[:-1]}"
        
    # cross validation dataset creation
    cv_generators = data_generator.create_cv_generators(pp_task_dir+'/',
                                                        num_cvs=len(subjects),
                                                        batch_size=opts['batch_size'], 
                                                        augment_list=((opts['rotation_x'], 
                                                                       opts['rotation_y'], 
                                                                       opts['rotation_z']), 
                                                                      opts['shift'], 
                                                                      opts['zoom']), 
                                                        mixed_batches=True)

    # Loop across each cross-validation split, run model training + evaluation and save in combined dictionaries
    for generators in cv_generators:    
        
        # pre-load the model
        (preload_model, preload_model_inference) = train.train_model(dataset=dataset_name, 
                                                                     generators=generators, 
                                                                     opts=opts, 
                                                                     return_untrained=True)
        preload_model.load_weights(opts['load_pretrained'] )
     
        (_,_,_,_,_,_,full_testing_list,_) = generators
        print(full_testing_list)
    
        # train the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            (model, model_inference) = train.train_model(dataset=dataset_name, 
                                                         generators=generators, 
                                                         opts=opts, 
                                                         use_multiprocessing=False,
                                                         return_untrained=False, 
                                                         verbose=1, 
                                                         save=False, 
                                                         model_path=model_dir, 
                                                         models=[preload_model, preload_model_inference])
    
		
        # evalutate training
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            (cv_evaluation, cv_scores) = train.evaluate_model(dataset=dataset_name,
                                                              model=model_inference, 
                                                              generators=generators,
                                                              save=False,
                                                              model_path=model_dir, 
                                                              verbose=1,
                                                              percentile_cut=80)
            # save scores and evaluation for each run
            evaluation = {**evaluation, **cv_evaluation}
            scores = {**scores, **cv_scores}

# Sava data
deepmreye_cor_xy = np.max(np.array(deepmreye_max_xy))/np.max(np.array(exp_max_xy))

for gaze_task in gaze_tasks: 
    # define filepath
    pp_task_dir = f"{pp_dir}/{gaze_task[:-1]}"

    for subject in subjects:
        pp_file = f"{pp_task_dir}/{subject}_{gaze_task}label.npz"
        mask_sub_dir = f"{mask_dir}/{subject}"
        pred_sub_dir =  f"{pred_dir}/{subject}"
        os.makedirs(pred_sub_dir, exist_ok=True)
    
        # get the list in the order as above
        subject_mask_files = []
        mask_files = glob.glob(f"{mask_sub_dir}/*{gaze_task}*.p")
        subject_mask_files.append(mask_files)
        subject_mask_files = [file for sublist in subject_mask_files for file in sublist]
        
        # get TR start and end for each runs
        values_array = np.array([num_gaze_centerfs if 'GazeCenterFS' in f
                                 else num_gaze_center_left_right 
                                 if 'GazeCenter_' in f 
                                 or 'GazeLeft_' in f 
                                 or'GazeRight_' in f 
                                 else 0 for f in subject_mask_files])
        second_column = np.zeros_like(values_array)
        second_column[1:] = np.cumsum(values_array[:-1])
        third_column = np.cumsum(values_array)
        mat_trs = np.column_stack((values_array, second_column, third_column))

        # get dataframes per runs
        for subject_mask_num, subject_mask_file in enumerate(subject_mask_files):
            
            df = adapt_evaluation(evaluation=evaluation,
                                  npz_fn=pp_file,
                                  cor_xy=deepmreye_cor_xy,
                                  TR_dur=TR,
                                  TR_start=mat_trs[subject_mask_num,1],
                                  TR_end=mat_trs[subject_mask_num,2],
                                  subTRs_num=subTRs)
            
            tsv_file = f"{pred_sub_dir}/{os.path.basename(subject_mask_file)[5:-2]}.tsv"
            print(tsv_file)
            df.to_csv(tsv_file, sep='\t', index=False)

# Chmod/chgrp
print(f"Changing files permissions in {sys.argv[1]}/{sys.argv[2]}")
os.system(f"chmod -Rf 771 {sys.argv[1]}/{sys.argv[2]}")
os.system(f"chgrp -Rf {sys.argv[4]} {sys.argv[1]}/{sys.argv[2]}")