function eyeLinkDrawLine(x,y,l_length,l_angle,color)
% ----------------------------------------------------------------------
% eyeLinkDrawLine(x,y,l_length,l_angle,color)
% ----------------------------------------------------------------------
% Goal of the function :
% Draw a line on the eyelink display.
% ----------------------------------------------------------------------
% Input(s) :
% x : center X 
% y : center Y 
% l_length : length of the line in pixels
% l_angle : angle of the line relative center (degrees)
% color : color of the line (1 to 15)
% ----------------------------------------------------------------------
% Output(s):
% none
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 21 / 11 / 2016
% Project :     pRF_gazeMod
% Version :     1.0
% ----------------------------------------------------------------------
x2 = x + cosd(l_angle) * l_length;
y2 = y - sind(l_angle) * l_length;
Eyelink('command','draw_line %d %d %d %d %d',round(x),round(y),round(x2),round(y2),color);

end