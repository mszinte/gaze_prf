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
python eye_analysis.py /scratch/mszinte/data gaze_prf 327 
-----------------------------------------------------------------------------------------
Written by Martin Szinte (martin.szinte@gmail.com)
-----------------------------------------------------------------------------------------
"""

import ipdb
import sys
import os
import glob
import warnings
import json
import numpy as np
import pandas as pd
from scipy.signal import detrend
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
import plotly.io as pio
import plotly.express as px

# Define paths to functional data
main_dir = f"{sys.argv[1]}/{sys.argv[2]}/derivatives/deepmreye"
pred_dir = f"{main_dir}/pred"
fig_dir = f"{main_dir}/figures"

# Make directories
os.makedirs(fig_dir, exist_ok=True)

# Define settings
with open('settings.json') as f:
    json_s = f.read()
    settings = json.loads(json_s)
subjects = settings['subjects']
sigma_smoothing = settings['sigma_smoothing']

# Postprocess data
for subject in subjects:
    pp_sub_dir = f"{pred_dir}/{subject}"
    eye_files = glob.glob(f"{pp_sub_dir}/*.tsv")
    for eye_file in eye_files:
        df = pd.read_csv(eye_file, sep='\t')

        print(eye_file)
        
        # linear detrending
        trend_x = np.polyfit(x=df.timestamps, y=df.x_coord, deg=1)
        trend_y = np.polyfit(x=df.timestamps, y=df.y_coord, deg=1)
        linear_trend_x = np.polyval(p=trend_x, x=df.timestamps)
        linear_trend_y = np.polyval(p=trend_y, x=df.timestamps)
        detrend_x = df.x_coord - linear_trend_x
        detrend_y = df.y_coord - linear_trend_y
        detrend_x += np.median(linear_trend_x)
        detrend_y += np.median(linear_trend_y)

        df['linear_trend_x'] = linear_trend_x
        df['x_coord_detrend'] = detrend_x
        df['y_coord_detrend'] = detrend_y

        # gaussian smoothing
        df['x_coord_detrend_gauss'] = gaussian_filter1d(df['x_coord_detrend'], sigma=sigma_smoothing)
        df['y_coord_detrend_gauss'] = gaussian_filter1d(df['y_coord_detrend'], sigma=sigma_smoothing)

        # get stats
        df['x_coord_detrend_gauss_mean'] = df['x_coord_detrend_gauss'].mean()
        df['y_coord_detrend_gauss_mean'] = df['y_coord_detrend_gauss'].mean()
        df['x_coord_detrend_gauss_std'] = df['x_coord_detrend_gauss'].std()
        df['y_coord_detrend_gauss_std'] = df['y_coord_detrend_gauss'].std()
        
        # Save the modified DataFrame back to the same file
        df.to_csv(eye_file, sep='\t', index=False)

# Settings
y_range = [-5, 5]
x_range = [-5, 5]
base_color = (0, 150, 175)  # Turquoise
row_titles = ["GazeCenterFS", "GazeCenter", "GazeLeft", "GazeRight"]

for subject in subjects:
    pp_sub_dir = f"{pred_dir}/{subject}"
    
    fig = make_subplots(rows=4, cols=3, 
                        vertical_spacing=0.1,
                        horizontal_spacing=0.11)
    
    for row in range(1,5):
        if row == 1: 
            eye_files = glob.glob(f"{pp_sub_dir}/*GazeCenterFS*.tsv")
        elif row == 2: 
            eye_files = glob.glob(f"{pp_sub_dir}/*GazeCenter_*.tsv")
        elif row == 3: 
            eye_files = glob.glob(f"{pp_sub_dir}/*GazeLeft_*.tsv")
        elif row == 4: 
            eye_files = glob.glob(f"{pp_sub_dir}/*GazeRight_*.tsv")

    
        for eye_file_num, eye_file in enumerate(eye_files):
    
            # Define colors
            opacity_values = np.linspace(0.2, 1, len(eye_files))
            colors = [f'rgba({base_color[0]}, {base_color[1]}, {base_color[2]}, {opacity})' for opacity in opacity_values]
        
            df = pd.read_csv(eye_file, sep='\t')
            fig.add_trace(go.Scatter(x=df['timestamps'], y=df['x_coord_detrend_gauss'], mode='lines', 
                                     line=dict(color=colors[eye_file_num])), row=row, col=1)
            fig.add_trace(go.Scatter(x=df['timestamps'], y=df['y_coord_detrend_gauss'], mode='lines', 
                                     line=dict(color=colors[eye_file_num])), row=row, col=2)
            fig.add_trace(go.Scatter(x=df['x_coord_detrend_gauss_mean'],
                                     y=df['y_coord_detrend_gauss_mean'],
                                     mode='markers', marker=dict(color=colors[eye_file_num], size=7), 
                                     error_x=dict(type='data', array=df['x_coord_detrend_gauss_std'], visible=True),
                                     error_y=dict(type='data', array=df['y_coord_detrend_gauss_std'], visible=True)),
                          row=row, col=3)
    
        
            for i in range(1, 3): fig.update_yaxes(range=y_range, row=row, col=i, zeroline=True, zerolinecolor='black', zerolinewidth=1) 
            fig.update_xaxes(range=x_range, row=row, col=3, zeroline=True, zerolinecolor='black', zerolinewidth=1)
            fig.update_yaxes(range=y_range, row=row, col=3, zeroline=True, zerolinecolor='black', zerolinewidth=1)
            
            # Set axis labels
            fig.update_xaxes(title_text='Time (sec)', row=row, col=1) 
            fig.update_xaxes(title_text='Time (sec)', row=row, col=2) 
            fig.update_yaxes(title_text=f"{row_titles[row-1]} X coord. (dva)", row=row, col=1) 
            fig.update_yaxes(title_text=f"{row_titles[row-1]} Y coord. (dva)", row=row, col=2) 
            fig.update_xaxes(title_text=f"{row_titles[row-1]} X coord. (dva)", row=row, col=3) 
            fig.update_yaxes(title_text=f"{row_titles[row-1]} Y coord. (dva)", row=row, col=3)
            
    fig.update_layout(showlegend=False, width=1100, height=1400, template="simple_white",)

    # Save figure
    fig_sub_dir = f"{fig_dir}/{subject}"
    os.makedirs(fig_sub_dir, exist_ok=True)
    print(f"{fig_sub_dir}/{subject}_deepmreye.pdf")
    pio.write_image(fig, f"{fig_sub_dir}/{subject}_deepmreye.pdf")

# Chmod/chgrp
print(f"Changing files permissions in {sys.argv[1]}/{sys.argv[2]}")
os.system(f"chmod -Rf 771 {sys.argv[1]}/{sys.argv[2]}")
os.system(f"chgrp -Rf {sys.argv[3]} {sys.argv[1]}/{sys.argv[2]}")