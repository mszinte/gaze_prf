function [expDes]=runSingleTrial(scr,const,expDes,my_key,t)
% ----------------------------------------------------------------------
% [expDes]=runSingleTrial(scr,const,expDes,my_key,t)
% ----------------------------------------------------------------------
% Goal of the function :
% Draw stimuli of each indivual trial and waiting for inputs
% ----------------------------------------------------------------------
% Input(s) :
% scr : struct containing screen configurations
% const : struct containing constant configurations
% expDes : struct containg experimental design
% my_key : structure containing keyboard configurations
% t : bar pass number
% ----------------------------------------------------------------------
% Output(s):
% resMat : experimental results (see below)
% expDes : struct containing all the variable design configurations.
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 01 / 06 / 2019
% Project :     pRFgazeMod
% Version :     4.0
% ----------------------------------------------------------------------

%% Compute and simplify var and rand
%  ---------------------------------

% trials number in this bar pass    
bar_trials              =   expDes.expMat(:,6) == t;
bar_trials_num          =   expDes.expMat(bar_trials,2);

% Cond1 : Discrimination task
cond1                   =   expDes.expMat(bar_trials,3);

% Cond2 : fixation direction
cond2                   =   expDes.expMat(bar_trials,4);

% Var 1 : Bar direction
var1                    =   expDes.expMat(bar_trials,5);

% Rand 1 : Fixation orientation
rand1                   =   expDes.expMat(bar_trials,8);

% Rand 2 : Stimulus orientation
rand2                   =   expDes.expMat(bar_trials,9);

if const.checkTrial && const.expStart == 0
    fprintf(1,'\n\n\t========================  BAR PASS %3.0f ========================\n',t);
    fprintf(1,'\n\tDiscrimnination task on      =\t');
    fprintf(1,'%s\t',expDes.txt_cond1{cond1(1)});
    fprintf(1,'\n\tFixation direction           =\t');
    fprintf(1,'%s\t',expDes.txt_cond2{cond2(1)});
    fprintf(1,'\n\tBar direction                =\t');
    fprintf(1,'%s\t',expDes.txt_var1{var1(1)});
    fprintf(1,'\n\tFixation orientation         =\t');
    fprintf(1,'%s\t',expDes.txt_rand1{rand1});
    fprintf(1,'\n\tStimulus orientation         =\t');
    fprintf(1,'%s\t',expDes.txt_rand2{rand2});
end

%% Prepare stimuli
%  ---------------
% Stimulus
if const.cond2 == 4
    if var1(1) == 1 || var1(1) == 5 || var1(1) == 9
        rand_num_tex_cond       =   const.rand_num_tex_hor;
        bar_step_val_cond       =   1:const.bar_step_hor;
        pre_txt_cond            =   'hor_';
        num_frame_max_cond      =   const.num_frame_max_hor;
        bar_step_cond           =   const.bar_steps_hor;
        time2load_cond          =   const.time2load_hor;
        time2draw_cond          =   const.time2draw_hor;
        time2probe_cond         =   const.time2probe_hor;
        time2resp_cond          =   const.time2resp_hor;
        time2make_cond          =   const.time2make_hor;
        resp_reset_cond         =   const.resp_reset_hor;
        time2log_cond           =   const.time2log_hor;
        trial_start_cond        =   const.trial_start_hor;
        trial_end_cond          =   const.trial_end_hor;
        probe_to_draw_cond      =   const.probe_to_draw_hor;
        frame_to_draw_cond      =   const.frame_to_draw_hor;
    elseif var1(1) == 3 || var1(1) == 7
        rand_num_tex_cond       =   const.rand_num_tex_ver;
        bar_step_val_cond       =   1:const.bar_step_ver;
        pre_txt_cond            =   'ver_';
        num_frame_max_cond      =   const.num_frame_max_ver;
        bar_step_cond           =   const.bar_steps_ver;
        time2load_cond          =   const.time2load_ver;
        time2draw_cond          =   const.time2draw_ver;
        time2probe_cond         =   const.time2probe_ver;
        time2resp_cond          =   const.time2resp_ver;
        time2make_cond          =   const.time2make_ver;
        resp_reset_cond         =   const.resp_reset_ver;
        time2log_cond           =   const.time2log_ver;
        trial_start_cond        =   const.trial_start_ver;
        trial_end_cond          =   const.trial_end_ver;
        probe_to_draw_cond      =   const.probe_to_draw_ver;
        frame_to_draw_cond      =   const.frame_to_draw_ver;
    end
else
    rand_num_tex_cond       =   const.rand_num_tex_apt;
    bar_step_val_cond       =   1:const.bar_step_apt;
    pre_txt_cond            =   'apt_';
    num_frame_max_cond      =   const.num_frame_max_apt;
    bar_step_cond           =   const.bar_steps_apt;
    time2load_cond          =   const.time2load_apt;
    time2draw_cond          =   const.time2draw_apt;
    time2probe_cond         =   const.time2probe_apt;
    time2resp_cond          =   const.time2resp_apt;
    time2make_cond          =   const.time2make_apt;
    resp_reset_cond         =   const.resp_reset_apt;
    time2log_cond           =   const.time2log_apt;
    trial_start_cond        =   const.trial_start_apt;
    trial_end_cond          =   const.trial_end_apt;
    probe_to_draw_cond      =   const.probe_to_draw_apt;
    frame_to_draw_cond      =   const.frame_to_draw_apt;
end

bar_step_val_rev_cond   =   bar_step_val_cond(end:-1:1);
rand_num_tex            =   rand_num_tex_cond;

num_tex                 =   1;
drawn_frame             =   0;
drawn_probe             =   0;
resp                    =   0;
missed_all              =   [];

% define image of next frame
bar_step                =   1;

% define order of the bar (only one direction is saved as img)
if var1(bar_step) == 1 || var1(bar_step) == 3
    bar_step_num            =   bar_step_val_cond(bar_step);
elseif var1(bar_step) == 5 || var1(bar_step) == 7
    bar_step_num            =   bar_step_val_rev_cond(bar_step);
elseif var1(bar_step) == 9
    bar_step_num            =  1:const.blk_step;
end

% define displayed direction of the bar (only 0 deg direcion motion
% reversing the bar and fix stim reversed)
if (var1(bar_step) == 3 || var1(bar_step) == 7) && const.cond2 ~= 4
    angle                   =   90;
    switch rand1(bar_step)
        case 1; fix_ori                 =   2;
        case 2; fix_ori                 =   1;
    end
    switch rand2(bar_step)
        case 1; stim_ori                =   2;
        case 2; stim_ori                =   1;
    end
else
    angle                   =   0;
    fix_ori                 =   rand1(bar_step);
    stim_ori                =   rand2(bar_step);
end

if var1(bar_step) == 9
    screen_filename         =   sprintf('%s/blank.mat',const.stim_folder);
    num_frame_max           =   const.num_frame_max_blk;
else
    if time2probe_cond(1,t)
        screen_filename         =   sprintf('%s/%sprobe_barStep-%i_kappaFix%i_kappaStim%i_fixOri-%i_stimOri-%i_noiseRand%i.mat',...
            const.stim_folder,pre_txt_cond,bar_step_num,expDes.fix_stair_val,expDes.stim_stair_val,fix_ori,stim_ori,rand_num_tex(num_tex));
    else
        screen_filename         =   sprintf('%s/%snoprobe_barStep-%i_noiseRand%i.mat',...
            const.stim_folder,pre_txt_cond,bar_step_num,rand_num_tex(num_tex));
    end
    num_frame_max           =   num_frame_max_cond;
end
load(screen_filename,'screen_stim');
expDes.texnew              =   Screen('MakeTexture',scr.main,screen_stim,[],[],[],angle);

% wait for T press in trial beginning
if t == 1
    % show the iti image
    time_start              =   GetSecs;
    expDes.tex              =   expDes.tex_blank;
    Screen('DrawTexture',scr.main,expDes.tex,[],const.stim_rect_cond);
    Screen('Flip',scr.main);
    tellapsed               =   GetSecs - time_start;
    first_tr                =   0;
    
    while ~first_tr
        if const.scanner == 0 || const.scannerTest
            WaitSecs(const.TR_dur-tellapsed);
            first_tr                =   1;
        else
            keyPressed              =   0;
            keyCode                 =   zeros(1,my_key.keyCodeNum);
            for keyb = 1:size(my_key.keyboard_idx,2)
                [keyP, keyC]            =   KbQueueCheck(my_key.keyboard_idx(keyb));
                keyPressed              =   keyPressed+keyP;
                keyCode                 =   keyCode+keyC;
            end
            if keyPressed
                if keyCode(my_key.escape) && const.expStart == 0
                    overDone(const,my_key)
                elseif keyCode(my_key.tr)
                    first_tr                =   1;
                end
            end
        end
    end
    
    % write in log/edf
    bar_pass_start          =   GetSecs;
    log_txt                 =   sprintf('bar pass %i event t at %f',t,bar_pass_start);
    if const.writeLogTxt
        fprintf(const.log_file_fid,'%s\n',log_txt);
    end
    if const.tracker
        Eyelink('message','%s',log_txt);
    end
end


%% Trial loop
%  ----------
nbf = 0;
while nbf < num_frame_max
    
    % flip count
	nbf                     =   nbf + 1;
    
    % define bar step position
    bar_step                =   bar_step_cond(nbf,t);
    
    % time to load the image
    time2load               =   time2load_cond(nbf,t);
    
    % time to make the texture
    time2make               =   time2make_cond(nbf,t);
    
    % define redraw time
    time2draw               =   time2draw_cond(nbf,t);
    
    % define probe frames
    time2probe              =   time2probe_cond(nbf,t);
    
    % define response frames
    time2resp               =   time2resp_cond(nbf,t);
    
    % log time
    time2log                =   time2log_cond(nbf,t);
    
    % define when to reset resp
    if resp_reset_cond(nbf,t)
        resp                =   0;
    end
    
    % load stim image
    if time2load
        % define order of the bar (only one direction is saved as img)
        if var1(bar_step) == 1 || var1(bar_step) == 3
            bar_step_num            =   bar_step_val_cond(bar_step);
        elseif var1(bar_step) == 5 || var1(bar_step) == 7
            bar_step_num            =   bar_step_val_rev_cond(bar_step);
        end
        
        % define displayed direction of the bar (only 0 deg direcion motion
        % reversing the bar and fix stim reversed)
        if (var1(bar_step) == 3 || var1(bar_step) == 7) && const.cond2 ~= 4
            angle                   =   90;
            switch rand1(bar_step)
                case 1; fix_ori             =   2;
                case 2; fix_ori             =   1;
            end
            switch rand2(bar_step)
                case 1; stim_ori            =   2;
                case 2; stim_ori            =   1;
            end
        else
            angle                   =   0;
            fix_ori                 =   rand1(bar_step);
            stim_ori                =   rand2(bar_step);
        end
        
        % define name of next frame
        if var1(bar_step) == 9
            screen_filename         =   sprintf('%s/blank.mat',const.stim_folder);
        else
            if time2probe
                screen_filename         =   sprintf('%s/%sprobe_barStep-%i_kappaFix%i_kappaStim%i_fixOri-%i_stimOri-%i_noiseRand%i.mat',...
                                                    const.stim_folder,pre_txt_cond,bar_step_num,expDes.fix_stair_val,expDes.stim_stair_val,...
                                                    fix_ori,stim_ori,rand_num_tex(num_tex));
            else
                screen_filename         =   sprintf('%s/%snoprobe_barStep-%i_noiseRand%i.mat',...
                                                    const.stim_folder,pre_txt_cond,bar_step_num,rand_num_tex(num_tex));
            end
        end
        % load the matrix
        load(screen_filename,'screen_stim');
    end
    
    % make the texture
    if time2make
        expDes.texnew           =   Screen('MakeTexture',scr.main,screen_stim,[],[],[],angle);
        
        % save fix and stim staircase level
        expDes.expMat(bar_trials_num(bar_step),12)  =   expDes.fix_stair_val;
        expDes.expMat(bar_trials_num(bar_step),13)  =   expDes.stim_stair_val;

        % define random number of noise patches
        if num_tex < size(const.rand_num_tex,2)
            num_tex                     =   num_tex + 1;
        end
    end
    
    % draw the texture
    if time2draw
        Screen('Close',expDes.tex);
        expDes.tex = expDes.texnew;
        tex2draw =   expDes.tex;
    end
    
    %% Screen flip
    Screen('DrawTexture',scr.main,tex2draw,[],const.stim_rect_cond)
    [~,~,~,missed]    =   Screen('Flip',scr.main);
    
    %% Screen flip
    %  -----------
    if sign(missed) == 1
        missed_val              =   1;
        missed_all              =   [missed_all;missed,missed_val];
    else
        missed_val              =   0;
        missed_all              =   [missed_all;missed,missed_val];
    end
    
    if time2draw
        drawn_frame             =   drawn_frame + 1 - missed_val;
        if time2probe
            drawn_probe             =   drawn_probe + 1 - missed_val;
        end
    end
    
    %% Create movie
    %  ------------
    if const.mkVideo
        expDes.vid_num          =   expDes.vid_num + 1;
        image_vid               =   Screen('GetImage', scr.main);
        imwrite(image_vid,sprintf('%s_frame_%i.png',const.movie_image_file,expDes.vid_num))
        writeVideo(const.vid_obj,image_vid);
    end

    %% Save trials times
    if time2log
        % probe onset
        if cond1(bar_step) == 1
            log_txt                 =   sprintf('bar pass %i fixation probe onset at %f',t,GetSecs);
        elseif cond1(bar_step) == 2
            log_txt                 =   sprintf('bar pass %i stimulus probe onset at %f',t,GetSecs);
        end
        if const.writeLogTxt
            fprintf(const.log_file_fid,'%s\n',log_txt);
        end
        if const.tracker
            Eyelink('message','%s',log_txt);
        end
        expDes.expMat(bar_trials_num(bar_step),15)  =   GetSecs;
    end    
    
    if trial_start_cond(nbf,t)
        % trial onset
        log_txt                 =   sprintf('bar pass %i trial onset at %f',t,GetSecs);
        if const.writeLogTxt
            fprintf(const.log_file_fid,'%s\n',log_txt);
        end
        if const.tracker
            Eyelink('message','%s',log_txt);
        end
        expDes.expMat(bar_trials_num(bar_step),10) =   GetSecs;
    end
    
    if trial_end_cond(nbf,t)
        % trial offset
        log_txt                 =   sprintf('bar pass %i trial offset at %f',t,GetSecs);
        if const.writeLogTxt
            fprintf(const.log_file_fid,'%s\n',log_txt);
        end
        if const.tracker
            Eyelink('message','%s',log_txt);
        end
        expDes.expMat(bar_trials_num(bar_step),11)  =   GetSecs;
    end
    
    %% Check keyboard
    %  --------------
    keyPressed              =   0;
    keyCode                 =   zeros(1,my_key.keyCodeNum);
    for keyb = 1:size(my_key.keyboard_idx,2)
        [keyP, keyC]            =   KbQueueCheck(my_key.keyboard_idx(keyb));
        keyPressed              =   keyPressed+keyP;
        keyCode                 =   keyCode+keyC;
    end
    
    if keyPressed
        if keyCode(my_key.tr)
            % write in log/edf
            log_txt                 =   sprintf('bar pass %i event t at %f',t,GetSecs);
            if const.writeLogTxt
                fprintf(const.log_file_fid,'%s\n',log_txt);
            end
            if const.tracker
                Eyelink('message','%s',log_txt);
            end
        end
        if keyCode(my_key.escape)
            if const.expStart == 0
                overDone(const,my_key)
            end
        elseif keyCode(my_key.left4)
            % update staircase
            if time2resp && resp == 0
                % write in log/edf
                log_txt                 =   sprintf('bar pass %i event %s at %f',t,my_key.left4Val,GetSecs);
                if const.writeLogTxt
                    fprintf(const.log_file_fid,'%s\n',log_txt);
                end
                if const.tracker
                    Eyelink('message','%s',log_txt);
                end
                
                if cond1(bar_step) == 1 % fixation task
                    switch rand1(bar_step)
                        case 1; response                =   0;
                        case 2; response                =   1;
                    end
                elseif cond1(bar_step) == 2 % stimuli task
                    switch rand2(bar_step)
                        case 1; response                =   0;
                        case 2; response                =   1;
                    end
                end
                expDes.expMat(bar_trials_num(bar_step),14)  =   response;
                expDes.expMat(bar_trials_num(bar_step),16)  =   GetSecs;
                [expDes]                =   updateStaircase(cond1(bar_step),const,expDes,response,GetSecs);
                resp                    =   1;
            end
        elseif keyCode(my_key.right1)  % cw button
            % update staircase
            if time2resp && resp == 0
                % write in log/edf
                log_txt                 =   sprintf('bar pass %i event %s at %f',t,my_key.right1Val,GetSecs);
                if const.writeLogTxt
                    fprintf(const.log_file_fid,'%s\n',log_txt);
                end
                if const.tracker
                    Eyelink('message','%s',log_txt);
                end
                if cond1(bar_step) == 1 % fixation task
                    switch rand1(bar_step)
                        case 1; response                =   1;
                        case 2; response                =   0;
                    end
                elseif cond1(bar_step) == 2 % stimuli task
                    switch rand2(bar_step)
                        case 1;	response                =   1;
                        case 2; response                =   0;
                    end
                end
                expDes.expMat(bar_trials_num(bar_step),14)  =   response;
                expDes.expMat(bar_trials_num(bar_step),16)  =   GetSecs;
                [expDes]                =   updateStaircase(cond1(bar_step),const,expDes,response,GetSecs);
                resp                    =   1;
            end 
        end
    end
 
    % dummy mode for scanner
    if const.scanner == 0 && const.scannerTest
        if mod(nbf,const.TR_num)==1
            % write in log/edf
            log_txt                 =   sprintf('bar pass %i event t at %f',t,GetSecs);
            if const.writeLogTxt
                fprintf(const.log_file_fid,'%s\n',log_txt);
            end
            if const.tracker
                Eyelink('message','%s',log_txt);
            end
        end
    end
end

%% Get number of stim and probe played
%  -----------------------------------
% write in log/edf
if var1(1) == 9
    probe_to_draw           =   0;
    frame_to_draw           =   const.frame_to_draw_blk;
else
    probe_to_draw           =   probe_to_draw_cond;
    frame_to_draw           =   frame_to_draw_cond;
end
log_txt                 =   sprintf('bar pass %i - %i/%i drawn fixation and bar stimuli - and %i/%i probes - %i missed sync on %i frames, %1.1f%% (mean/median delay = %1.1f/%1.1f ms)',...
                                            t,drawn_frame,frame_to_draw,drawn_probe,probe_to_draw,...
                                            [sum(missed_all(:,2)>0),size(missed_all,1),sum(missed_all(:,2)>0)/size(missed_all,1)*100],...
                                            mean(missed_all(missed_all(:,2)==1))*1000,median(missed_all(missed_all(:,2)==1))*1000);
if const.writeLogTxt
    fprintf(const.log_file_fid,'%s\n',log_txt);
end
if const.tracker
    Eyelink('message','%s',log_txt);
end

end