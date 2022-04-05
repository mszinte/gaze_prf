function [const]=dirSaveFile(const)
% ----------------------------------------------------------------------
% [const]=dirSaveFile(const)
% ----------------------------------------------------------------------
% Goal of the function :
% Make directory and saving files name and fid.
% ----------------------------------------------------------------------
% Input(s) :
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Output(s):
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 01 / 06 / 2019
% Project :     pRFgazeMod
% Version :     4.0
% ----------------------------------------------------------------------

% Create data directory 
if ~isdir(sprintf('data/%s/%s/func/',const.sjct,const.sesNumTxt))
    mkdir(sprintf('data/%s/%s/func/',const.sjct,const.sesNumTxt))
end
% Define directory
const.dat_output_file   =   sprintf('data/%s/%s/func/%s_%s_task-%s%s_run-%i',const.sjct,const.sesNumTxt,const.sjct,const.sesNumTxt,const.cond1_txt,const.cond2_txt,const.cond_run_num(const.runNum));

% Eye data
const.eyelink_temp_file =   'XX.edf';
const.eyelink_local_file=   sprintf('%s_eyeData.edf',const.dat_output_file);

% Behavioral data
const.behav_file        =   sprintf('%s_events.tsv',const.dat_output_file);
if const.expStart
    if exist(const.behav_file,'file')
        aswErase = upper(strtrim(input(sprintf('\n\tThis file allready exist, do you want to erase it ? (Y or N): '),'s')));
        if upper(aswErase) == 'N'
            error('Please restart the program with correct input.')
        elseif upper(aswErase) == 'Y'
        else
            error('Incorrect input => Please restart the program with correct input.')
        end
    end
end
const.behav_file_fid    =   fopen(const.behav_file,'w');

% Log file
if const.writeLogTxt
    const.log_file          =   sprintf('%s_logData.txt',const.dat_output_file);
    const.log_file_fid      =   fopen(const.log_file,'w');
end

% Create additional info directory
if ~isdir(sprintf('data/%s/add/',const.sjct))
    mkdir(sprintf('data/%s/add/',const.sjct))
end

% Define directory
const.add_output_file   =   sprintf('data/%s/add/%s_%s_task-%s%s_run-%i',const.sjct,const.sjct,const.sesNumTxt,const.cond1_txt,const.cond2_txt,const.cond_run_num(const.runNum));

% Define .mat saving file
const.mat_file          =   sprintf('%s_matFile.mat',const.add_output_file);

% Staircase file
const.staircase_file    =   sprintf('data/%s/add/%s_staircases.mat',const.sjct,const.sjct);

% Define .mat stimuli file
const.stim_folder       =   sprintf('stim/screenshots');

% Movie file
if const.mkVideo
    if ~isdir(sprintf('others/%s_vid/',const.cond2_txt))
        mkdir(sprintf('others/%s_vid/',const.cond2_txt))
    end
    const.movie_image_file  =   sprintf('others/%s_vid/%s_vid',const.cond2_txt,const.cond2_txt);
    const.movie_file        =   sprintf('others/%s_vid.mp4',const.cond2_txt);
end

end