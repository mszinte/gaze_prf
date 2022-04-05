# GAZE_PRF

By : Martin Szinte & Gilles de Hollander<br/>
With : Marco Aqil, Serge Dumoulin & Tomas Knapen<br/>

## Experiment description
Experiment in which we first used a full screen 4 direction (left/right/up/down)
bar pass stimuli in a attetion to fixation or attention to the bar experiment.
Next, we use the same task but this time using a bar pass restricted to an aperture and 
displayed at 3 different position surrounding the fixation target put at the screen center 
or displaced to the left or to the right.<br/>

## Experiment codes
* 8 participants tested using __experiment_codes/main/expLauncher.m__

## Analysis codes

### Behavioral analysis


### MRI analysis

# Pre-processing
* Convert data in bids.<br/>
* Run fmriprpep with anat-only option using _analysis_codes/preproc/fmriprep_sbatch.py_<br/>
* Manual edition of the pial surface using freeview launched using _analysis_codes/preproc/pial_edits.py_ and following these [rules](http://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/PialEditsV6.0)<br/>
* Re-run freesurfer with change of the pial surface using _analysis_codes/preproc/freesurfer_pial.py_<br/>
* Cut brains and flatten hemispheres using _analysis_codes/preproc/flatten_sbatch.py_<br/>
* Run fmriprpep for functionnal runs using _analysis_codes/preproc/fmriprep_sbatch.py_<br/>
* Deface T1w/T2w data using _analysis_codes/preproc/deface_sbatch.py_<br/>
* Run pybest to z-scores, high pass filter the data using _analysis_codes/preproc/pybest_sbatch.py_<br/>
* Arrange data in pp_data folder _analysis_codes/preproc/preproc_end.py_<br/>
* Import in pycortex surfaces and flatmaps using _analysis_codes/preproc/pycortex_import.py_<br/>



