function imageMatCol = compBarAperture(const)
% ----------------------------------------------------------------------
% imageMatCol = compBarAperture(const)
% ----------------------------------------------------------------------
% Goal of the function :
% Compute bar aperture of the stimulus
% ----------------------------------------------------------------------
% Input(s) :
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Output(s):
% imageMatCol: 3D-RGBA matrix of the bar aperture
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 01 / 06 / 2018
% Project :     pRF_exp
% Version :     3.0
% ----------------------------------------------------------------------
% 1 - opaque
% 0 = transparent
raised_cos              =   const.raised_cos;
cRf                     =   const.aperture_blur*const.stim_size(1)*2;
raised_cos              =   raised_cos(1:end-1);
raised_cos              =   imresize(raised_cos,round([cRf,1]),'bilinear');
fall_cos                =   flipud(raised_cos);

bar_mask_size           =   const.bar_mask_size;
imageMat                =   ones(round(bar_mask_size),1);


imageMat(round(bar_mask_size/2 - const.bar_width/2) - size(raised_cos,1):round(bar_mask_size/2 - const.bar_width/2)-1) = fall_cos'; 
imageMat(round(bar_mask_size/2 - const.bar_width/2):round(bar_mask_size/2 + const.bar_width/2)) = 0; 
imageMat(round(bar_mask_size/2 + const.bar_width/2)+1:round(bar_mask_size/2 + const.bar_width/2)+size(raised_cos,1)) = raised_cos'; 
imageMat                =   repmat(imageMat,1,round(bar_mask_size));

imageMatCol(:,:,1)      =   imageMat*0+const.background_color(1);
imageMatCol(:,:,2)      =   imageMat*0+const.background_color(2);
imageMatCol(:,:,3)      =   imageMat*0+const.background_color(3);
imageMatCol(:,:,4)      =   imageMat*255;

end