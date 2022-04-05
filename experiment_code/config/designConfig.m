function [expDes]=designConfig(const)
% ----------------------------------------------------------------------
% [expDes]=designConfig(const)
% ----------------------------------------------------------------------
% Goal of the function :
% Define experimental design
% ----------------------------------------------------------------------
% Input(s) :
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Output(s):
% expDes : struct containg experimental design
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 01 / 06 / 2019
% Project :     pRFgazeMod
% Version :     4.0
% ----------------------------------------------------------------------

%% Experimental random variables

% Cond 1 : task (2 modalities)
% =======
expDes.oneC             =   [1;2];
expDes.txt_cond1        =   {'fixation','stimulus'};
% 01 = task on fixation stimulus
% 02 = task on bar stimulus

% Cond 2 : fixation direction (3 modalities)
% =======
expDes.twoC             =   [1;2;3;4];
expDes.txt_cond2        =   {'left','center','right','centerFS'};
% 01 = fixation left
% 02 = fixation center
% 03 = fixation right
% 04 = fixation right full screen

% Var 1 : bar direction (9 modalities)
% ======
expDes.oneV             =   [1;3;5;7];
expDes.txt_var1         =   {'180 deg','225 deg','270 deg','315 deg','0 deg','45 deg','90 deg','135 deg','none'};
% value defined as const.subdirection in constConfig.m
% 01 = 180 deg
% 02 = 225 deg
% 03 = 270 deg
% 04 = 315 deg
% 05 =   0 deg
% 06 =  45 deg
% 07 =  90 deg
% 08 = 135 deg
% 09 = none

% Rand 1: fixation orientation (2 modalities)
% =======
expDes.oneR             =   [1;2];
expDes.txt_rand1        =   {'cw','ccw','none'};
% 01 = tilt cw
% 02 = tilt ccw
% 03 = none

% Rand 2: bar stimulus orientation (3 modalities)
% =======
expDes.twoR             =   [1;2];
expDes.txt_rand2        =   {'cw','ccw','none'};
% 01 = tilt cw
% 02 = tilt ccw
% 03 = none

% Staircase
% ---------
if const.runNum == 1 && const.sesNum == 1
    % create staircase starting value
    expDes.fix_stair_val    =   const.fix_stair_val;
    expDes.cor_count_fix    =   0;
    expDes.incor_count_fix  =   0;
    expDes.stim_stair_val   =   const.stim_stair_val;
    expDes.cor_count_stim   =   0;
    expDes.incor_count_stim =   0;
else
    % load staircase of previous blocks
    load(const.staircase_file);
    expDes.fix_stair_val    =   staircase.fix_stair_val;
    expDes.cor_count_fix    =   staircase.cor_count_fix;
    expDes.incor_count_fix  =   staircase.incor_count_fix;
    expDes.stim_stair_val   =   staircase.stim_stair_val;
    expDes.cor_count_stim   =   staircase.cor_count_stim;
    expDes.incor_count_stim =   staircase.incor_count_stim;
end

%% Experimental configuration :
expDes.nb_cond          =   2;
expDes.nb_var           =   1;
expDes.nb_rand          =   2;
expDes.nb_list          =   0;

%% Experimental loop
rng('default');rng('shuffle');
runT                    =   const.runNum;

switch const.cond2 
    case 1;bar_dir           =   const.bar_hor_dir_run;
    case 2;bar_dir           =   const.bar_hor_dir_run;
    case 3;bar_dir           =   const.bar_hor_dir_run;
	case 4;bar_dir           =   const.bar_all_dir_run;
end

t_trial = 0;
for t_bar_pass = 1:size(bar_dir,2)
    cond1                       =   const.cond1;
    cond2                       =   const.cond2;
    rand_var1                   =   bar_dir(t_bar_pass);
    
    if rand_var1 == 9
        bar_pos_per_pass  =   const.blk_step;
    else
        if const.cond2 == 4
            if rand_var1 == 1 || rand_var1 == 5
                bar_pos_per_pass     =   const.bar_step_hor;
            elseif rand_var1 == 3 || rand_var1 == 7
                bar_pos_per_pass     =   const.bar_step_ver;
            end
        else
            bar_pos_per_pass     =   const.bar_step_apt;
        end
    end
    
    for bar_step = 1:bar_pos_per_pass
    
        rand_rand1                  =   expDes.oneR(randperm(numel(expDes.oneR),1));
        rand_rand2                  =   expDes.twoR(randperm(numel(expDes.twoR),1));

        % no bar
        if rand_var1 == 9
            rand_var1                   =   9;
            rand_rand1                  =   3;
            rand_rand2                  =   3;
        end
    
        t_trial     =   t_trial + 1;
        
        expDes.expMat(t_trial,:)=   [   runT,           t_trial,        cond1,          cond2,          rand_var1,  ...
                                        t_bar_pass,     bar_step,       rand_rand1,     rand_rand2,     NaN,        ...
                                        NaN,            NaN,            NaN,            NaN,            NaN,        ...
                                        NaN];
        % col 01:   Run number
        % col 02:   Trial number
        % col 03:   Attention task
        % col 04:   Fixation direction
        % col 05:   Bar direction
        % col 06:   Bar pass period
        % col 07:   Bar step
        % col 08:   Fixation noise orientation
        % col 09:   Stimulus noise orientation
        % col 10:   Trial onset time
        % col 11:   Trial offset time
        % col 12:   Fixation noise staircase value
        % col 13:   Stimulus noise staircase value
        % col 14:   Reponse value (correct/incorrect)
        % col 15:   Probe time
        % col 16:   Response time
    end
end

expDes.nb_trials = size(expDes.expMat,1);

end