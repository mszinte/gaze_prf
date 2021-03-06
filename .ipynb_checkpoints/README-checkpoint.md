# GAZE_PRF

By : Martin Szinte & Gilles de Hollander<br/>
With : Marco Aqil, Serge Dumoulin & Tomas Knapen<br/>

## Experiment description
Experiment in which we first used a full screen 4 direction (left/right/up/down)
bar pass stimuli in a attention to fixation or attention to the bar experiment.
Next, we use the same tasks but this time using a bar pass restricted to an aperture and 
displayed at 3 different position surrounding the fixation target put at the screen center 
or displaced to the left or to the right.<br/>

## Experiment code
* 8 participants tested using _experiment_code/main/expLauncher.m_

## Analysis code

### Behavioral analysis
* Compute and plot performance results using _analysis_code/behav/behav_results.ipynb_<br/>
* Compute and plot eccentricity results using _analysis_code/behav/ecc_results.ipynb_<br/>

### MRI analysis

# Pre-processing
* Convert data in bids.<br/>
* Run fmriprpep with anat-only option using _analysis_code/preproc/fmriprep_sbatch.py_<br/>
* Manual edition of the pial surface using freeview launched using _analysis_code/preproc/pial_edits.py_ and following these [rules](http://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/PialEditsV6.0)<br/>
* Re-run freesurfer with change of the pial surface using _analysis_code/preproc/freesurfer_pial.py_<br/>
* Cut brains and flatten hemispheres using _analysis_code/preproc/flatten_sbatch.py_<br/>
* Run fmriprpep for functionnal runs using _analysis_code/preproc/fmriprep_sbatch.py_<br/>
* Deface T1w/T2w data using _analysis_codes/preproc/deface_sbatch.py_<br/>
* Run pybest to z-scores, high pass filter the data using _analysis_code/preproc/pybest_sbatch.py_<br/>
* Arrange data in pp_data folder _analysis_code/preproc/preproc_end.py_<br/>
* Import in pycortex surfaces and flatmaps using _analysis_code/preproc/pycortex_import.py_<br/>
* Average runs togehter _analysis_code/preproc/average_runs.py_<br/>

# Post-processing
* Compute pRF across conditions using _GILLES_ ...
* Create pRF threshold mask using _analysis_code/prf/prf_th_masks.ipynb_<br/>
* Generate Fullscreen retinotopy maps _analysis_code/prf/pycortex.ipynb_<br/>
* Draw ROIS using Inkscape and Fullscreen maps<br/>
* Define ROI masks nifti files using _analysis_code/prf/roi_masks.ipynb_<br/>
* Generate "all" pycortex flatmaps and webgl using _analysis_code/prf/pycortex.ipynb_<br/>
* Push subjects webgl online using _analysis_code/prf/webgl.ipynb_<br/>
* Create TSV files of stats comparisons using _analysis_code/prf/make_tsv.ipynb_<br/>
* Compute pickle files with all timeseries/predictions of out of set analysis _analysis_code/prf/make_tsv.ipynb_<br/>
* Draw Fullscreen attention R2 comparison using _analysis_code/prf/attcmp_plots.ipynb_<br/>
* Draw exemple V1 timeseries and pRF model using _analysis_code/prf/timeseries_plots.ipynb_<br/>
* Compute out of set r2 change using _analysis_code/prf/fs_fit_cmp_plots.ipynb_<br/>
* Draw refit pRFx parameter using _analysis_code/prf/refit_pRFx_plots.ipynb_<br/>
* Draw refit reference frame index using _analysis_code/prf/refit_indexcmp_plots.ipynb_<br/>
