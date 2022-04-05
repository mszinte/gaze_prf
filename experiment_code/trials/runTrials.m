function [const] = runTrials(scr,const,expDes,el,my_key)
% ----------------------------------------------------------------------
% [const] = runTrials(scr,const,expDes,el,my_key)
% ----------------------------------------------------------------------
% Goal of the function :
% Launch each trial, instructions and connection with eyelink
% ----------------------------------------------------------------------
% Input(s) :
% scr : struct containing screen configurations
% const : struct containing constant configurations
% expDes : struct containg experimental design
% my_key : structure containing keyboard configurations
% ----------------------------------------------------------------------
% Output(s):
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 01 / 06 / 2019
% Project :     pRFgazeMod
% Version :     4.0
% ----------------------------------------------------------------------

% Configuration of videos
% -----------------------
if const.mkVideo
    expDes.vid_num          =   0;
    const.vid_obj           =   VideoWriter(const.movie_file,'MPEG-4');
    const.vid_obj.FrameRate =   60;
	const.vid_obj.Quality   =   100;
    open(const.vid_obj);
end

% Special instruction for scanner
% -------------------------------
if const.scanner && ~const.scannerTest
    scanTxt                 =   '_Scanner';
else
    scanTxt                 =   '';
end

% Save all config at start of the block
% -------------------------------------
config.scr              =   scr;
config.const            =   const;
config.expDes           =   expDes;
config.my_key           =   my_key;
save(const.mat_file,'config');

% First mouse config
% ------------------
if const.expStart
    HideCursor;
    for keyb = 1:size(my_key.keyboard_idx,2)
        KbQueueFlush(my_key.keyboard_idx(keyb));
    end
end

% Initial calibrations
% --------------------
if const.tracker
    fprintf(1,'\tEye tracking instructions - press space or 1st-red left button box -\n');
    eyeLinkClearScreen(el.bgCol);
    eyeLinkDrawText(scr.x_mid,scr.y_mid,el.txtCol,'CALIBRATION INSTRUCTION - PRESS SPACE');
    instructionsIm(scr,const,my_key,sprintf('Calibration%s',scanTxt),0);
    calibresult             =   EyelinkDoTrackerSetup(el);
    if calibresult == el.TERMINATE_KEY
        return
    end
end

for keyb = 1:size(my_key.keyboard_idx,2)
    KbQueueFlush(my_key.keyboard_idx(keyb));
end

% Start Eyelink
% -------------
record                  =   0;
while ~record
    if const.tracker
        if ~record
            Eyelink('startrecording');
            key                     =   1;
            while key ~=  0
                key                     =   EyelinkGetKey(el);
            end
            error                   =   Eyelink('checkrecording');
            if error==0
                record                  =   1;
                Eyelink('message', 'RECORD_START');
            else
                record                  =   0;
                Eyelink('message', 'RECORD_FAILURE');
            end
        end
    else
        record                  =   1;
    end
end


% Task instructions 
fprintf(1,'\n\tTask instructions - press space or 1st-red left button box -');
if const.tracker
    eyeLinkClearScreen(el.bgCol);
    eyeLinkDrawText(scr.x_mid,scr.y_mid,el.txtCol,'TASK INSTRUCTIONS - PRESS SPACE')
end
instructionsIm(scr,const,my_key,sprintf('%s%s',const.cond1_txt,scanTxt),0);
for keyb = 1:size(my_key.keyboard_idx,2)
    KbQueueFlush(my_key.keyboard_idx(keyb));
end
fprintf(1,'\n\n\tBUTTON PRESSED BY SUBJECT\n');

% make empty texture
screen_filename         =   sprintf('%s/blank.mat',const.stim_folder);
load(screen_filename,'screen_stim');
expDes.tex_blank        =   Screen('MakeTexture',scr.main,screen_stim);

% Main Loop
% ---------
for t = 1:const.bar_dir_num
    
    % Write on eyelink screen
    if const.tracker
        drawTrialInfoEL(scr,const,t)
    end
    
    % Write in log/edf
    log_txt                     =   sprintf('bar pass %i started at %f\n',t,GetSecs);
    if const.writeLogTxt
        fprintf(const.log_file_fid,log_txt);
    end
    if const.tracker
        Eyelink('message','%s',log_txt);
        Eyelink('command', 'record_status_message ''BAR PASS %d''', t);
    end
    
    % Run single trial
    [expDes]                    =   runSingleTrial(scr,const,expDes,my_key,t);
    
    % write in log/edf
    log_txt                     =   sprintf('bar pass %i stopped at %f',t,GetSecs);
    if const.writeLogTxt
        fprintf(const.log_file_fid,'%s\n',log_txt);
    end
    if const.tracker
        Eyelink('message', '%s',log_txt);
    end
end

% Compute/Write mean/std behavioral data
% --------------------------------------
behav_txt_head{1}       =   'onset';                        behav_mat_res{1}        =   expDes.expMat(:,10);
behav_txt_head{2}       =   'duration';                     behav_mat_res{2}        = 	expDes.expMat(:,11)-expDes.expMat(:,10);
behav_txt_head{3}       =   'run_number';                   behav_mat_res{3}        = 	expDes.expMat(:,1);
behav_txt_head{4}       =   'trial_number';                 behav_mat_res{4}        = 	expDes.expMat(:,2);
behav_txt_head{5}       =   'att_task';                     behav_mat_res{5}        = 	expDes.expMat(:,3);
behav_txt_head{6}       =   'fix_direction';                behav_mat_res{6}        = 	expDes.expMat(:,4);
behav_txt_head{7}       =   'bar_direction';                behav_mat_res{7}        = 	expDes.expMat(:,5);
behav_txt_head{8}       =   'bar_period';                   behav_mat_res{8}        = 	expDes.expMat(:,6);
behav_txt_head{9}       =   'bar_step';                     behav_mat_res{9}        = 	expDes.expMat(:,7);
behav_txt_head{10}      =   'fix_noise_ori';                behav_mat_res{10}       = 	expDes.expMat(:,8);
behav_txt_head{11}      =   'stim_noise_ori';               behav_mat_res{11}       = 	expDes.expMat(:,9);
behav_txt_head{12}      =   'fix_stair_val';                behav_mat_res{12}       = 	expDes.expMat(:,12);
behav_txt_head{13}      =   'stim_stair_val';               behav_mat_res{13}       = 	expDes.expMat(:,13);
behav_txt_head{14}      =   'response_val';                 behav_mat_res{14}       = 	expDes.expMat(:,14);
behav_txt_head{15}      =   'probe_time';                   behav_mat_res{15}       = 	expDes.expMat(:,15);
behav_txt_head{16}      =   'reaction_time';                behav_mat_res{16}       = 	expDes.expMat(:,16)-expDes.expMat(:,15);

head_line               =   [];
for trial = 1:expDes.nb_trials
    % header line
    if trial == 1
        for tab = 1:size(behav_txt_head,2)
            if tab == size(behav_txt_head,2)
                head_line               =   [head_line,sprintf('%s',behav_txt_head{tab})];
            else
                head_line               =   [head_line,sprintf('%s\t',behav_txt_head{tab})];
            end
        end
        fprintf(const.behav_file_fid,'%s\n',head_line);
    end
    
	% trials line
    trial_line              =   [];
    for tab = 1:size(behav_mat_res,2)
        if tab == size(behav_mat_res,2)
            if isnan(behav_mat_res{tab}(trial))
                trial_line              =   [trial_line,sprintf('n/a')];
            else
                trial_line              =   [trial_line,sprintf('%1.10g',behav_mat_res{tab}(trial))];
            end
        else
            if isnan(behav_mat_res{tab}(trial))
                trial_line              =   [trial_line,sprintf('n/a\t')];
            else
                trial_line              =   [trial_line,sprintf('%1.10g\t',behav_mat_res{tab}(trial))];
            end
        end
    end
    fprintf(const.behav_file_fid,'%s\n',trial_line);
end

% End messages
% ------------
if const.runNum == size(const.cond_run_order,1)
    instructionsIm(scr,const,my_key,'End',1);
else
    instructionsIm(scr,const,my_key,'End_block',1);
end

% Save all config at the end of the block (overwrite start made at start)
% ---------------------------------------
config.scr = scr; config.const = const; config.expDes = expDes; config.my_key = my_key;
save(const.mat_file,'config');

% Save staircases
% ---------------
staircase.fix_stair_val     =   expDes.fix_stair_val;
staircase.cor_count_fix     =   expDes.cor_count_fix;
staircase.incor_count_fix   =   expDes.incor_count_fix;
staircase.stim_stair_val    =   expDes.stim_stair_val;
staircase.cor_count_stim    =   expDes.cor_count_fix;
staircase.incor_count_stim  =   expDes.incor_count_fix;
save(const.staircase_file,'staircase');

% Stop Eyelink
% ------------
if const.tracker
    Eyelink('command','clear_screen');
    Eyelink('command', 'record_status_message ''END''');
    WaitSecs(1);
    Eyelink('stoprecording');
    Eyelink('message', 'RECORD_STOP');
    eyeLinkClearScreen(el.bgCol);eyeLinkDrawText(scr.x_mid,scr.y_mid,el.txtCol,'THE END - PRESS SPACE OR WAIT');
end

end