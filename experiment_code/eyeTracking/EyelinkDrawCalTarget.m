function rect=EyelinkDrawCalTarget(el, x, y)
% ----------------------------------------------------------------------
% rect=EyelinkDrawCalTarget(el, x, y)
% ----------------------------------------------------------------------
% Goal of the function :
% Modification of the EyeLink toolbox to draw target for calibration
% adjusted by a specified values measure in constConfig and to draw dots
% with anti-aliasing.
% ----------------------------------------------------------------------
% Input(s) :
% wPtr : window pointer.
% const : struct containing many constant configuration.
% ----------------------------------------------------------------------
% Output(s):
% el : eye-link structure.
% error : disconnected link code. 
% const : struct containing edfFileName
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 02 / 06 / 2018
% Project :     pRF_gazeMod
% Version :     3.0
% ----------------------------------------------------------------------

Screen('FillRect',el.window,el.backgroundcolour);
Screen('DrawDots', el.window,[x,y],el.fixation_outer_rim_rad*2,el.fixation_outer_rim_color,[],2);
Screen('DrawDots', el.window,[x,y],el.fixation_rim_rad *2,el.fixation_rim_color ,[],2);
Screen('DrawDots', el.window,[x,y],el.fixation_rad*2,el.fixation_color,[],2);
Screen( 'Flip',  el.window);

rect = round([x-el.fixation_rad, y-el.fixation_rad, x+el.fixation_rad, y+el.fixation_rad]);

end