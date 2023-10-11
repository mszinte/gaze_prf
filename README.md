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
* Compute and plot performance results: [_behav_results.ipynb_](analysis_code/behav/behav_results.ipynb)<br/>
* Compute and plot eccentricity results using [_ecc_results.ipynb_](analysis_code/behav/ecc_results.ipynb)<br/>

### MRI analysis

#### Pre-processing
* Convert data in bids.<br/>
* Run fmriprpep with anat-only option: [_fmriprep_sbatch.py_](analysis_code/preproc/fmriprep_sbatch.py)<br/>
* Manual edition of the pial surface using freeview launched: [_pial_edits.py_](analysis_code/preproc/pial_edits.py) and following these [rules](http://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/PialEditsV6.0)<br/>
* Re-run freesurfer with change of the pial surface: [_freesurfer_pial.py_](analysis_code/preproc/freesurfer_pial.py)<br/>
* Cut brains and flatten hemispheres: [_flatten_sbatch.py_](analysis_code/preproc/flatten_sbatch.py)<br/>
* Run fmriprpep for functionnal runs: [_fmriprep_sbatch.py_](analysis_code/preproc/fmriprep_sbatch.py)<br/>
* Deface T1w/T2w data: [_deface_sbatch.py_](analysis_code/preproc/deface_sbatch.py)<br/>
* Run pybest to z-scores, high pass filter the data: [_pybest_sbatch.py_](analysis_code/preproc/pybest_sbatch.py)<br/>
* Arrange data in pp_data folder: [_preproc_end.py_](analysis_code/preproc/preproc_end.py)<br/>
* Import in pycortex surfaces and flatmaps: [_pycortex_import.py_](analysis_code/preproc/pycortex_import.py)<br/>
* Average runs together: [_average_runs.py_](analysis_code/preproc/average_runs.py)<br/>

##### Post-processing
* Compute pRF across conditions 
* Compute pRF out-of_set fit
* Compute pRF refit
* Create pRF threshold mask: [_prf_th_masks.ipynb_](analysis_code/prf/prf_th_masks.ipynb)<br/>
* Generate Fullscreen retinotopy maps: [_pycortex.ipynb_](analysis_code/prf/pycortex.ipynb)<br/>
* Draw ROIS using Inkscape and Fullscreen maps<br/>
* Define ROI masks nifti files: [_roi_masks.ipynb_](analysis_code/prf/roi_masks.ipynb)<br/>
* Generate "all" pycortex flatmaps and webgl: [_pycortex.ipynb_](analysis_code/prf/pycortex.ipynb)<br/>
* Push subjects webgl online: [_webgl.ipynb_](analysis_code/prf/webgl.ipynb)<br/>
* Create TSV files of stats comparisons: [_make_tsv.ipynb_](analysis_code/prf/make_tsv.ipynb)<br/>
* Compute pickle files with all timeseries/predictions of out of set analysis: [_make_tsv.ipynb_](analysis_code/prf/make_tsv.ipynb)<br/>
* Draw Fullscreen attention R2 comparison: [_attcmp_plots.ipynb_](analysis_code/prf/attcmp_plots.ipynb)<br/>
* Draw timeseries and pRF model: [_timeseries_plot.ipynb_](analysis_code/prf/timeseries_plot.ipynb)<br/>
* Compute out of set r2 change: [_fs_fit_cmp_plots.ipynb_](analysis_code/prf/fs_fit_cmp_plots.ipynb)<br/>
* Draw refit pRFx parameter: [_refit_pRFx_plots.ipynb_](analysis_code/prf/refit_pRFx_plots.ipynb)<br/>
* Draw refit reference frame index: [_refit_indexcmp_plots.ipynb_](analysis_code/prf/refit_indexcmp_plots.ipynb)<br/>
* Compute decoding outcomes pickle files: [_make_tsv.ipynb_](analysis_code/prf/make_tsv.ipynb)<br/>
* Draw decoding time series using [_decode_timeseries_plot.ipynb_](analysis_code/decode/decode_timeseries_plot.ipynb)<br/>
* Draw decoding time series across bar pass: [_decode_time_cor_plot.ipynb_](analysis_code/decode/decode_time_cor_plot.ipynb)<br/>
* Draw decoding correlations to ground truth: [_decode_correlation_plot.ipynb_](analysis_code/decode/decode_correlation_plot.ipynb)<br/>
* Draw decoding reference frame index: [_decode_ref_index_plot.ipynb_](analysis_code/decode/decode_ref_index_plot.ipynb)<br/>
* Compute decoding statistics [_Central_stats_decoding.ipynb.ipynb_](analysis_code/decode/Central\stats\decoding.ipynb)<br/>
* Statistics for manuscript [_manuscript_stats.ipynb_](analysis_code/manuscript/manuscript_stats.ipynb)<br/>

_Optional:_
* Compute GLM in link with gainfield results [_get_gainfield_betas.py_](analysis_code/glm/get_gainfield_betas.py)<br/>
* Draw pycortex flatmap of gainfield results [_pycortex.ipynb_](analysis_code/glm/pycortex.ipynb)<br/>
* Draw GLM with gainfiled results comparison [_glmcmp_plots.ipynb_](analysis_code/glm/glmcmp_plots.ipynb)<br/>

#### Figures

##### Figure 1
* Figure 1C: [_behav_results.ipynb_](analysis_code/behav/behav_results.ipynb)<br/>
* Figure 1D: [_timeseries_plot.ipynb_](analysis_code/prf/timeseries_plot.ipynb)<br/>
* Figure 1G-H: [_pycortex.ipynb_](analysis_code/prf/pycortex.ipynb)<br/>
* Figure 1I: [_attcmp_plots.ipynb_](analysis_code/prf/attcmp_plots.ipynb)<br/>

##### Figure 2
* Figure 2B-C: [_timeseries_plot.ipynb_](analysis_code/prf/timeseries_plot.ipynb)<br/>
* Figure 2D-E: [_fs_fit_cmp_plots.ipynb_](analysis_code/prf/fs_fit_cmp_plots.ipynb)<br/>

##### Figure 3
* Figure 3C-D: [_refit_pRFx_plots.ipynb_](analysis_code/prf/refit_pRFx_plots.ipynb)<br/>
* Figure 3E: [_refit_indexcmp_plots.ipynb_](analysis_code/prf/refit_indexcmp_plots.ipynb)<br/>
* Figure 3F-G: [_pycortex.ipynb_](analysis_code/prf/pycortex.ipynb)<br/>

##### Figure 4
* Figure 4A-B: [_timeseries_plot.ipynb_](analysis_code/prf/timeseries_plot.ipynb)<br/>
* Figure 4C: [_decode_timeseries_plot.ipynb_](analysis_code/decode/decode_timeseries_plot.ipynb)<br/>
* Figure 4D-E: [_decode_time_cor_plot.ipynb_](analysis_code/decode/decode_time_cor_plot.ipynb)<br/>
* Figure 4F: [_decode_correlation_plot.ipynb_](analysis_code/decode/decode_correlation_plot.ipynb)<br/>
* Figure 4G: [_decode_ref_index_plot.ipynb_](analysis_code/decode/decode_ref_index_plot.ipynb)<br/>