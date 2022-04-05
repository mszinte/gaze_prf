%% General experimenter launcher
%  =============================
% By :      Martin SZINTE
% Projet :  pRF_gazeMod experiment
% With :    Tomas KNAPEN
% Version:  4.0

% Version description
% ===================
% Experiment in which we first used a full screen 4 direction (left/right/up/down)
% bar pass stimuli in a attetion to fixation or attention to the bar experiment.
% Next, we use the same task but this time using a bar pass restricted to an aperture and 
% displayed at 3 different position surrounding the fixation target put at the screen center 
% or displaced to the left or to the right.

% To run
% ======

% 1. change screen to take out the mirror display mode
%    a. click NVIDIA X Server Settings
%    b. in X Server Display Configuration, in selection pick select BOLD screen
%    c. in configuration put new X screen
%    d. set resolution to 1920x1080 and 120Hz
%    e. click on apply
%    f. and click save to X configuration file + type password + quit + restart computer 

% 2. open matlab by typing in terminal "sudo /home/lab/Desktop/matlab" + type password (watdanook)

% 3. run first or second sessions
%    a. check below that const.expStart = 1 & const.scannerTest = 0 & const.scanner = 1
%    b. click on run in expLauncher.m (!!!! select add top path when asked   !!!!)
%    c. follow instructions (select participant number, session, run)
%    d. to quit instruction press space (or ask subject to press button indicated when ready) the code wait first TR
%    e. do 10 runs per session, 2 session per subjects

% 4. push the data online 
%    a. in terminal: cd /home/lab/Experiments/pRFgazeMod/
%    b. in terminal: git add -A
%    c. in terminal: git commit -m "data of sub-001"
%    d. in terminal: git push origin master

% 5. put back the screen in mirror display mode
%    a. click NVIDIA X Server Settings
%    b. in X Server Display Configuration, in selection pick select BOLD screen
%    c. in configuration put X screen 0
%    d. click on apply
%    e. click save to X configuration file + type password + quit
%    f. switch off and on the computer, and check that screen settings back to mirror display

% TO DO
% =====
% code analysis of behavioral data
% code analysis of mri data

% First settings
% --------------
Screen('CloseAll');clear all;clear mex;clear functions;close all;home;ListenChar(1);AssertOpenGL;

% General settings
% ----------------
const.expName           =   'pRFgazeMod';   % experiment name.
const.expStart          =   1;              % Start of a recording exp                          0 = NO  , 1 = YES
const.checkTrial        =   0;              % Print trial conditions (for debugging)            0 = NO  , 1 = YES
const.writeLogTxt       =   1;              % write a log file in addition to eyelink file      0 = NO  , 1 = YES
const.genStimuli        =   0;              % Generate all stimuli                              0 = NO  , 1 = YES
const.drawStimuli       =   0;              % Draw stimuli generated                            0 = NO  , 1 = YES
const.mkVideo           =   0;              % Make a video of a run (on mac not linux)          0 = NO  , 1 = YES

% External controls
% -----------------
const.tracker           =   0;              % run with eye tracker                              0 = NO  , 1 = YES
const.scanner           =   1;              % run in MRI scanner                                0 = NO  , 1 = YES
const.scannerTest       =   0;              % run with T returned at TR time                    0 = NO  , 1 = YES

% Durations
% ---------
% 2 SESSIONS OF 35 min each of run + topup
% AttendFixGazeCenterFS:  8 x 3:00 min (150 TR)
% AttendFixGazeLeft:      4 x 2:26 min (122 TR)
% AttendFixGazeRight:     4 x 2:26 min (122 TR)
% AttendFixGazeCenter:    4 x 2:26 min (122 TR)

% Run order
% ---------
const.cond_run_order_ses1 = [1,4;2,4;...     %    run 01 - AttendFixGazeCenterFS_run1  | run 02 - AttendStimGazeCenterFS_run1 |  L1+R1+U1+D1
                             1,4;2,4;...     %    run 03 - AttendFixGazeCenterFS_run2  | run 04 - AttendStimGazeCenterFS_run2 |  L2+R2+U2+D2
                             1,1;2,1;...     %    run 05 - AttendFixGazeLeft_run1      | run 06 - AttendStimGazeLeft_run1     |  L2+R2
                             1,3;2,3;...     %    run 07 - AttendFixGazeRight_run1     | run 08 - AttendStimGazeRight_run1    |  L2+R2
                             1,2;2,2];       %    run 09 - AttendFixGazeCenter_run1    | run 10 - AttendStimGazeCenter_run1   |  L2+R2                             
                             
const.cond_run_order_ses2 = [1,4;2,4;...     %    run 01 - AttendFixGazeCenterFS_run1  | run 02 - AttendStimGazeCenterFS_run1 |  L3+R3+U3+D3
                             1,4;2,4;...     %    run 03 - AttendFixGazeCenterFS_run2  | run 04 - AttendStimGazeCenterFS_run2 |  L4+R4+U4+D4
                             1,1;2,1;...     %    run 05 - AttendFixGazeLeft_run1      | run 06 - AttendStimGazeLeft_run1     |  L4+R4
                             1,3;2,3;...     %    run 07 - AttendFixGazeRight_run1     | run 08 - AttendStimGazeRight_run1    |  L4+R4
                             1,2;2,2];       %    run 09 - AttendFixGazeCenter_run1    | run 10 - AttendStimGazeCenter_run1   |  L4+R4                             

% Run number per condition
% ------------------------
const.cond_run_num_ses1 =  [1;1;...
                            2;2;...
                            1;1;...
                            1;1;...
                            1;1];
                        
const.cond_run_num_ses2 =  [1;1;...
                            2;2;...
                            1;1;...
                            1;1;...
                            1;1];
                            
% Desired screen setting
% ----------------------
const.desiredFD         =   120;            % Desired refresh rate
%fprintf(1,'\n\n\tDon''t forget to change before testing\n');
const.desiredRes        =   [1920,1080];    % Desired resolution

% Path
% ----
dir                     =   (which('expLauncher'));
cd(dir(1:end-18));

% Add Matlab path
% ---------------
addpath('config','main','conversion','eyeTracking','instructions','trials','stim');

% Subject configuration
% ---------------------
[const]                 =   sbjConfig(const);
                        
% Main run:
% ---------
main(const);