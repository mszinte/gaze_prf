function drawTrialInfoEL(scr,const,t)
% ----------------------------------------------------------------------
% drawTrialInfoEL(scr,const,t)
% ----------------------------------------------------------------------
% Goal of the function :
% Draw on the eyelink display the trial configuration
% ----------------------------------------------------------------------
% Input(s) :
% scr : struct containing screen configurations
% const : struct containing constant configurations
% expDes : struct containg experimental design
% t : trial number
% ----------------------------------------------------------------------
% Output(s):
% none
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 05 / 12 / 2018
% Project :     pRF_gazeMod
% Version :     4.0
% ----------------------------------------------------------------------

% o--------------------------------------------------------------------o
% | EL Color index                                                     |
% o----o----------------------------o----------------------------------o
% | Nb |  Other(cross,box,line)     | Clear screen                     |
% o----o----------------------------o----------------------------------o
% |  0 | black                      | black                            |
% o----o----------------------------o----------------------------------o
% |  1 | dark blue                  | dark dark blue                   |
% o----o----------------------------o----------------------------------o
% |  2 | dark green                 | dark blue                        |
% o----o----------------------------o----------------------------------o
% |  3 | dark turquoise             | blue                             |
% o----o----------------------------o----------------------------------o
% |  4 | dark red                   | light blue                       |
% o----o----------------------------o----------------------------------o
% |  5 | dark purple                | light light blue                 |
% o----o----------------------------o----------------------------------o
% |  6 | dark yellow (brown)        | turquoise                        |
% o----o----------------------------o----------------------------------o
% |  7 | light gray                 | light turquoise                  | 
% o----o----------------------------o----------------------------------o
% |  8 | dark gray                  | flashy blue                      |
% o----o----------------------------o----------------------------------o
% |  9 | light purple               | green                            |
% o----o----------------------------o----------------------------------o
% | 10 | light green                | dark dark green                  |
% o----o----------------------------o----------------------------------o
% | 11 | light turquoise            | dark green                       |
% o----o----------------------------o----------------------------------o
% | 12 | light red (orange)         | green                            |
% o----o----------------------------o----------------------------------o
% | 13 | pink                       | light green                      |
% o----o----------------------------o----------------------------------o
% | 14 | light yellow               | light green                      |
% o----o----------------------------o----------------------------------o
% | 15 | white                      | flashy green                     |
% o----o----------------------------o----------------------------------o

% Color config
frameCol                =   15;
ftCol                   =   15;
textCol                 =   15;
bgCol                   =   0;
stimCol                 =   4;

% Clear screen
eyeLinkClearScreen(bgCol);

% Cond2 : fixation direction
cond2                   =   const.cond2;
rect_ctr                =   [scr.x_mid,scr.y_mid] + const.stim_offset(cond2,:);

%% Draw Stimulus
% Fixation box
eyeLinkDrawBox(rect_ctr(1),rect_ctr(2),const.fix_out_rim_rad*2,const.fix_out_rim_rad*2,2,frameCol,ftCol);

% Two lines of text during trial (slow process)
text0                   =   sprintf('Cond 1 = %s | Cond 2 = %s | BarPassNum = %i',const.cond1_txt,const.cond2_txt,t);
eyeLinkDrawText(scr.x_mid,scr.scr_sizeY - 30,textCol,text0);
WaitSecs(0.1);

end