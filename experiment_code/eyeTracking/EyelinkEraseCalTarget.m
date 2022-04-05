function erasecaltarget(el, rect)

% erase calibration target
%
% USAGE: erasecaltarget(el, rect)
%
%		el: eyelink default values
%		rect: rect that will be filled with background colour 
if ~IsEmptyRect(rect)
    Screen( 'FillRect',  el.window, el.backgroundcolour );	% clear_cal_display()
    Screen( 'Flip',  el.window);
end
