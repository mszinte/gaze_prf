function main(const)
% ----------------------------------------------------------------------
% main(const)
% ----------------------------------------------------------------------
% Goal of the function :
% Launch all function of the experiment
% ----------------------------------------------------------------------
% Input(s) :
% const : struct containing a lot of constant configuration
% ----------------------------------------------------------------------
% Output(s):
% none
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 01 / 06 / 2019
% Project :     pRFgazeMod
% Version :     4.0
% ----------------------------------------------------------------------

% File director
% -------------
[const]                 =   dirSaveFile(const);

% Screen configurations
% ---------------------
[scr]                   =   scrConfig(const);

% Keyboard configurations
% -----------------------
tic;
[my_key]                =   keyConfig;

% Experimental constant
% ---------------------
[const]                 =   constConfig(scr,const);

% Experimental design
% -------------------
[expDes]                =   designConfig(const);

% Open screen window
% ------------------
[scr.main,scr.rect]     =   Screen('OpenWindow',scr.scr_num,const.background_color,[], scr.clr_depth,2);
[~]                     =   Screen('BlendFunction', scr.main, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
priorityLevel           =   MaxPriority(scr.main);Priority(priorityLevel);

% Initialise eye tracker
% ----------------------
el                      =   [];
if const.tracker
    [el]                    =   initEyeLink(scr,const,my_key);
end

% Draw and save all textures
% --------------------------
if const.genStimuli
    makeTextures(scr,const,expDes);
end

% Trial runner
% ------------
ListenChar(2);
[const]                 =   runTrials(scr,const,expDes,el,my_key);

% End
% ---
overDone(const,my_key)

end