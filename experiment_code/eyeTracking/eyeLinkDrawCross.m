function eyeLinkDrawCross(x,y)
% ----------------------------------------------------------------------
% eyeLinkDrawCross(x,y)
% ----------------------------------------------------------------------
% Goal of the function :
% Draw a small cross a a specific location on the eyelink display.
% ----------------------------------------------------------------------
% Input(s) :
% x : center X of the small cross
% y : center Y of the small cross
% ----------------------------------------------------------------------
% Output(s):
% none
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 21 / 11 / 2016
% Project :     pRF_gazeMod
% Version :     1.0
% ----------------------------------------------------------------------

Eyelink('command','draw_cross %d %d', round(x),round(y));

end