function eyeLinkDrawBox(x,y,sizeX,sizeY,ELflag,colorFrm,colorFill)
% ----------------------------------------------------------------------
% eyeLinkDrawBox(x,y,sizeX,sizeY,ELflag,colorFrm,colorFill)
% ----------------------------------------------------------------------
% Goal of the function :
% Draw a box on the eyelink display.
% ----------------------------------------------------------------------
% Input(s) :
% x : center X 
% y : center Y 
% size X : horrizontal size of the box
% size Y : vertical size of the box
% ELFlag : if 1 : framed box
%          if 2 : filled box
%          if 3 : filled framed box
% colorFrm : color of the frame
% colorFill : color of the box
% ----------------------------------------------------------------------
% Output(s):
% none
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 21 / 11 / 2016
% Project :     pRF_gazeMod
% Version :     1.0
% ----------------------------------------------------------------------

if ELflag == 1 % framed box
    Eyelink('Command','draw_box %d %d %d %d %d',round(x-sizeX/2), round(y-sizeY/2), round(x+sizeX/2), round(y+sizeY/2),colorFrm);
elseif ELflag == 2
    Eyelink('Command','draw_filled_box %d %d %d %d %d',round(x-sizeX/2), round(y-sizeY/2), round(x+sizeX/2), round(y+sizeY/2),colorFill);
elseif ELflag == 3
    Eyelink('Command','draw_filled_box %d %d %d %d %d',round(x-sizeX/2), round(y-sizeY/2), round(x+sizeX/2), round(y+sizeY/2),colorFill);
    Eyelink('Command','draw_box %d %d %d %d %d',round(x-sizeX/2), round(y-sizeY/2), round(x+sizeX/2), round(y+sizeY/2),colorFrm);
end
    
end