function [const]=constConfig(scr,const)
% ----------------------------------------------------------------------
% [const]=constConfig(scr,const)
% ----------------------------------------------------------------------
% Goal of the function :
% Define all constant configurations
% ----------------------------------------------------------------------
% Input(s) :
% scr : struct containing screen configurations
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Output(s):
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 01 / 06 / 2019
% Project :     pRFgazeMod
% Version :     4.0
% ----------------------------------------------------------------------

% Randomization
rng('default');
rng('shuffle');

%% Colors;
const.white             =   [255,255,255];                                                      % white color
const.black             =   [0,0,0];                                                            % black color
const.gray              =   [128,128,128];                                                      % gray color
const.background_color  =   const.black;                                                        % background color
const.stim_color        =   const.white;                                                        % stimulus color
const.ann_color         =   const.stim_color;                                                   % define anulus around fixation color
const.ann_probe_color   =   const.stim_color;                                                   % define anulus around fixation color when probe
const.dot_color         =   const.stim_color;                                                   % define fixation dot color
const.dot_probe_color   =   const.black;                                                        % define fixation dot color when probe

%% Time parameters
const.TR_dur            =   1.3;                                                                % repetition time
const.TR_num            =   (round(const.TR_dur/scr.frame_duration));                           % repetition time in screen frames
const.bar_dir_num       =   9;                                                                  % number of bar passes and break

const.bar_step_ver      =   18;                                                                 % bar steps for vertical bar pass
const.bar_step_hor      =   32;                                                                 % bar steps for horizontal bar pass 
const.bar_step_apt      =   18;                                                                 % bar steps for apperture bar pass
const.blk_step          =   10;                                                                 % blank period step

const.bar_step_dur_ver  =   const.TR_dur;                                                       % bar step duration for vertical bar pass in seconds
const.bar_step_num_ver  =   (round(const.bar_step_dur_ver/scr.frame_duration));                 % bar step duration for vertical bar pass in screen frames
const.bar_step_dur_hor  =   const.TR_dur;                                                       % bar step duration for horizontal bar pass in seconds
const.bar_step_num_hor  =   (round(const.bar_step_dur_hor/scr.frame_duration));                 % bar step duration for horizontal bar pass in screen frames
const.bar_step_dur_apt  =   const.TR_dur;                                                       % bar step duration for apperture bar pass in seconds
const.bar_step_num_apt  =   (round(const.bar_step_dur_apt/scr.frame_duration));                 % bar step duration for apperture bar pass in screen frames
const.blk_step_dur      =   const.TR_dur;                                                       % blank step duration in seconds
const.blk_step_num      =   (round(const.blk_step_dur/scr.frame_duration));                     % blank step duration in screen frames

const.noise_freq        =   10;                                                                 % compute noise frequency in hertz
const.patch_dur         =   1/const.noise_freq;                                                 % compute single patch duration in seconds
const.patch_num         =   (round(const.patch_dur/scr.frame_duration));                        % compute single patch duration in screen frames

const.probe_duration    =   0.600;                                                              % probe duration in seconds
const.probe_num_redraw  =   (round(const.probe_duration/(1/const.noise_freq)));                 % probe duration in redraw frames
const.probe_num         =   (round(const.probe_duration/scr.frame_duration));                   % probe duration in screen frames

const.frame_to_draw_ver =   const.bar_step_ver*const.bar_step_dur_ver*const.noise_freq;         % number of drawn frame per pass for vertical bar pass
const.frame_to_draw_hor =   const.bar_step_hor*const.bar_step_dur_hor*const.noise_freq;         % number of drawn frame per pass for horizontal bar pass
const.frame_to_draw_apt =   const.bar_step_apt*const.bar_step_dur_apt*const.noise_freq;         % number of drawn frame per pass for horizontal bar pass
const.frame_to_draw_blk =   const.blk_step*const.TR_dur*const.noise_freq;                       % number of drawn frame per bar pass for blank

const.probe_to_draw_ver =   const.bar_step_ver*const.probe_num_redraw;                          % number of probes to draw per pass in horizontal bar pass
const.probe_to_draw_hor =   const.bar_step_hor*const.probe_num_redraw;                          % number of probes to draw per pass in vertical bar pass
const.probe_to_draw_apt =   const.bar_step_apt*const.probe_num_redraw;                          % number of probes to draw per pass in aperture bar pass
const.num_frame_max_hor =   const.bar_step_hor*const.TR_num;                                    % number of flip per pass in horizontal bar pass
const.num_frame_max_ver =   const.bar_step_ver*const.TR_num;                                    % number of flip per pass in vertical bar pass
const.num_frame_max_apt =   const.bar_step_apt*const.TR_num;                                    % number of flip per pass in aperture bar pass
const.num_frame_max_blk =   const.blk_step*const.TR_num;                                        % number of flip per pass when blank bar pass

%% Stim parameters
% Noise patches
const.noise_num         =   5;                                                                  % number of generated patches per kappa
const.stim_size         =   [scr.scr_sizeX/2,scr.scr_sizeY/2];                                  % full screen stimuli size in pixels
const.apt_rad_val       =   4;                                                                  % aperture stimuli radius in dva
const.apt_rad           =   vaDeg2pix(const.apt_rad_val,scr);                                   % aperture stimuli radius in pixels

const.stim_offset_val   =   [-4,0;0,0;+4,0;0,0];                                                % stimulus x/y offset in degrees in pRF and colormatcher task
const.stim_offset       =   vaDeg2pix(const.stim_offset_val,scr);                               % stimulus x/y offset in pixels in pRF and colormatcher task
const.stim_rect         =   [   scr.x_mid-const.stim_size(1);...                                % rect of the actual stimulus
                                scr.y_mid-const.stim_size(2);...
                                scr.x_mid+const.stim_size(1);...
                                scr.y_mid+const.stim_size(2)];
                            
const.stim_rect_cond    =   const.stim_rect + [ const.stim_offset(const.cond2,:)';...           % rect of the actual stimulus in the specific condition
                                                const.stim_offset(const.cond2,:)'];
const.num_steps_kappa   =   15;                                                                 % number of kappa steps
const.noise_kappa       =   [0,10.^(linspace(-1,1.5,const.num_steps_kappa-1))];                 % von misses filter kappa parameter (1st = noise, last = less noisy)
const.good_4_harder     =   3;                                                                  % amount of trials before (harder) staircase update
const.bad_4_easier      =   1;                                                                  % amount of trials before (easier) staircase update
const.fix_stair_val     =   round(const.num_steps_kappa*0.60);                                  % starting value of the fixation staircase kappa value
const.stim_stair_val    =   round(const.num_steps_kappa*0.60);                                  % starting value of the stimulus staircase kappa value
if const.mkVideo
    const.fix_stair_val     =   const.num_steps_kappa;                                          % starting value of the fixation staircase kappa value
    const.stim_stair_val    =   const.num_steps_kappa;                                          % starting value of the stimulus staircase kappa value
end
const.noise_size        =   sqrt((const.stim_size(1)*2)^2+(const.stim_size(1)*2)^2);            % size of the patch to allow 45deg rotation
const.noise_angle       =   [45,-45,NaN];                                                       % noise rotation angles
const.noise_pixelVal    =   0.04;                                                               % stimulus noise pixel size in degrees
const.noise_pixel       =   vaDeg2pix(const.noise_pixelVal,scr);                                % stimulus noise pixel size in pixels
const.native_noise_dim  =   round([const.noise_size/const.noise_pixel,const.noise_size/const.noise_pixel]);% starting size of the patch
const.noise_color       =   'pink';                                                             % stimuli noise color ('white','pink','brownian');

% compute random image order
const.rand_num_tex      =   [];
for t = 1:ceil((const.bar_step_hor*const.TR_dur*const.noise_freq)/const.noise_num)
    new_rand_num_tex        =   randperm(const.noise_num);
    if t > 1
        while new_rand_num_tex(1) == rand_num_tex(end)
            new_rand_num_tex    =   randperm(const.noise_num);
        end
    end
    rand_num_tex            =   new_rand_num_tex;
    const.rand_num_tex       =   [const.rand_num_tex,rand_num_tex];
end
const.rand_num_tex_ver  =   const.rand_num_tex(1:round(const.frame_to_draw_ver));
const.rand_num_tex_hor  =   const.rand_num_tex(1:round(const.frame_to_draw_hor));
const.rand_num_tex_apt  =   const.rand_num_tex(1:round(const.frame_to_draw_apt));
const.rand_num_tex_blk  =   const.rand_num_tex(1:round(const.frame_to_draw_blk));

const.rect_noise        =  [scr.x_mid - const.noise_size/2;...                                  % noise native rect
                            scr.y_mid - const.noise_size/2;...
                            scr.x_mid + const.noise_size/2;...
                            scr.y_mid + const.noise_size/2];

% Apertures
const.rCosine_grain     =   80;                                                                 % grain of the radius cosine steps
const.aperture_blur     =   0.01;                                                               % ratio of the apperture that is blured following an raised cosine
const.raised_cos        =   cos(linspace(-pi,pi,const.rCosine_grain*2));
const.raised_cos        =   const.raised_cos(1:const.rCosine_grain)';                           % cut half to have raising cosinus function
const.raised_cos        =   (const.raised_cos - min(const.raised_cos))/...
                                    (max(const.raised_cos)-min(const.raised_cos));              % normalize raising cosinus function

% Stimulus circular apertures
const.stim_aperture     =   compStimAperture(const);                                            % define stimulus aperture alpha layer

% Fixation circular aperture
const.fix_out_rim_radVal=   0.4;                                                                % radius of outer circle of fixation bull's eye
const.fix_rim_radVal    =   0.9*const.fix_out_rim_radVal;                                       % radius of intermediate circle of fixation bull's eye in degree
const.fix_radVal        =   0.1*const.fix_out_rim_radVal;                                       % radius of inner circle of fixation bull's eye in degrees
const.fix_out_rim_rad   =   vaDeg2pix(const.fix_out_rim_radVal,scr);                            % radius of outer circle of fixation bull's eye in pixels
const.fix_rim_rad       =   vaDeg2pix(const.fix_rim_radVal,scr);                                % radius of intermediate circle of fixation bull's eye in pixels
const.fix_rad           =   vaDeg2pix(const.fix_radVal,scr);                                    % radius of inner circle of fixation bull's eye in pixels
const.fix_aperture      =   compFixAperture(const);                                             % define fixation aperture
const.fix_annulus       =   compFixAnnulus(const);
const.fix_dot           =   compFixDot(const);
const.fix_dot_probe     =   const.fix_dot;

% Bar
const.bar_all_dir_run   =   [9,1,9,3,9,5,9,7,9];                                                % direction (1 = 180 deg, 2 = 225 deg, 3 =  270 deg, 4 = 315 deg,
                                                                                                %           5 = 0 deg, 6 = 45 deg, 7 = 90 deg, 8 = 135 deg; 9 = none)
const.bar_hor_dir_run   =   [9,1,9,5,9,1,9,5,9];                                                % direction (1 = 180 deg, 2 = 225 deg, 3 =  270 deg, 4 = 315 deg,
                                                                                                %           5 = 0 deg, 6 = 45 deg, 7 = 90 deg, 8 = 135 deg; 9 = none)

const.bar_width_deg     =   1;                                                                  % bar width in dva
const.bar_width         =   vaDeg2pix(const.bar_width_deg,scr);                                 % bar width in pixels

const.bar_mask_size     =   const.stim_size(1)*4;                                               % bar mask size in pixels
const.bar_aperture      =   compBarAperture(const);                                             % compute bar aperture mesh

const.bar_dir_ang_start =   0;                                                                  % first direction angle
const.bar_step_dir_ang  =   45;                                                                 % step between direction angles
const.bar_dir_ang       =   const.bar_dir_ang_start:const.bar_step_dir_ang:(360-const.bar_step_dir_ang+const.bar_dir_ang_start);

const.bar_aperture_rect =  [scr.x_mid - const.bar_mask_size/2,...                               % bar native rect
                            scr.y_mid - const.bar_mask_size/2,...
                            scr.x_mid + const.bar_mask_size/2,...
                            scr.y_mid + const.bar_mask_size/2];
const.bar_step_size_ver =   const.stim_size(2)*2/(const.bar_step_ver-1);                        % bar step size in pixels for vertical bar pass
const.bar_step_size_hor =   const.stim_size(1)*2/(const.bar_step_hor-1);                        % bar step size in pixels for horizontal bar pass
const.bar_step_size_apt =   const.apt_rad*2/(const.bar_step_apt-1);                             % bar step size in pixels for aperture bar pass

% define horizontal bar pass
for tAng = 1:size(const.bar_dir_ang,2)
    for tSteps = 1:const.bar_step_hor
        
        const.barCtr_hor(1,tSteps,tAng) =  scr.x_mid + (cosd(const.bar_dir_ang(tAng)) * const.stim_size(1))...
                                                     - (cosd(const.bar_dir_ang(tAng)) * const.bar_step_size_hor*(tSteps-1));    % bar center coord x
        const.barCtr_hor(2,tSteps,tAng) =  scr.y_mid + (-sind(const.bar_dir_ang(tAng)) * const.stim_size(2))...
                                                     - (-sind(const.bar_dir_ang(tAng)) * const.bar_step_size_hor*(tSteps-1));   % bar center coord y
        
        const.bar_aperture_rect_hor(:,:,tSteps,tAng) =  CenterRectOnPoint(const.bar_aperture_rect,const.barCtr_hor(1,tSteps,tAng),const.barCtr_hor(2,tSteps,tAng));                 % bar rect
        const.bar_aperture_rect_hor(:,:,tSteps,tAng) =  const.bar_aperture_rect_hor(:,:,tSteps,tAng);% + [const.stim_offset(const.cond2,:),const.stim_offset(const.cond2,:)];
        
        const.probe_num_start_hor(tSteps)  =    const.bar_step_num_hor*(tSteps-1) + const.bar_step_num_hor/2 - const.probe_num/2;      % bar probe frame start
        const.probe_num_end_hor(tSteps)    =    const.bar_step_num_hor*(tSteps-1) + const.bar_step_num_hor/2 + const.probe_num/2 - 1;  % bar probe frame end
        const.resp_num_start_hor(tSteps)   =    const.probe_num_start_hor(tSteps);
        const.resp_num_end_hor(tSteps)     =    const.resp_num_start_hor(tSteps)+const.bar_step_num_hor - 1;
    end
end

% define vertical bar pass
for tAng = 1:size(const.bar_dir_ang,2)
    for tSteps = 1:const.bar_step_ver
        
        const.barCtr_ver(1,tSteps,tAng) =  scr.x_mid + (cosd(const.bar_dir_ang(tAng)) * const.stim_size(1))...
                                                     - (cosd(const.bar_dir_ang(tAng)) * const.bar_step_size_ver*(tSteps-1));    % bar center coord x
        const.barCtr_ver(2,tSteps,tAng) =  scr.y_mid + (-sind(const.bar_dir_ang(tAng)) * const.stim_size(2))...
                                                     - (-sind(const.bar_dir_ang(tAng)) * const.bar_step_size_ver*(tSteps-1));   % bar center coord y
        
        const.bar_aperture_rect_ver(:,:,tSteps,tAng) =  CenterRectOnPoint(const.bar_aperture_rect,const.barCtr_ver(1,tSteps,tAng),const.barCtr_ver(2,tSteps,tAng));     % bar rect
        const.bar_aperture_rect_ver(:,:,tSteps,tAng) =  const.bar_aperture_rect_ver(:,:,tSteps,tAng);% + [const.stim_offset(const.cond2,:),const.stim_offset(const.cond2,:)];
        
        const.probe_num_start_ver(tSteps)  =    const.bar_step_num_ver*(tSteps-1) + const.bar_step_num_ver/2 - const.probe_num/2;      % bar probe frame start
        const.probe_num_end_ver(tSteps)    =    const.bar_step_num_ver*(tSteps-1) + const.bar_step_num_ver/2 + const.probe_num/2 - 1;  % bar probe frame end
        const.resp_num_start_ver(tSteps)   =    const.probe_num_start_ver(tSteps);
        const.resp_num_end_ver(tSteps)     =    const.resp_num_start_ver(tSteps)+const.bar_step_num_ver - 1;
    end
end

% define apeture bar pass
for tAng = 1:size(const.bar_dir_ang,2)
    for tSteps = 1:const.bar_step_ver
        
        const.barCtr_apt(1,tSteps,tAng) =  scr.x_mid + (cosd(const.bar_dir_ang(tAng)) * const.apt_rad)...
                                                     - (cosd(const.bar_dir_ang(tAng)) * const.bar_step_size_apt*(tSteps-1));    % bar center coord x
        const.barCtr_apt(2,tSteps,tAng) =  scr.y_mid + (-sind(const.bar_dir_ang(tAng)) * const.apt_rad)...
                                                     - (-sind(const.bar_dir_ang(tAng)) * const.bar_step_size_apt*(tSteps-1));   % bar center coord y
        
        const.bar_aperture_rect_apt(:,:,tSteps,tAng) =  CenterRectOnPoint(const.bar_aperture_rect,const.barCtr_apt(1,tSteps,tAng),const.barCtr_apt(2,tSteps,tAng));     % bar rect
        const.bar_aperture_rect_apt(:,:,tSteps,tAng) =  const.bar_aperture_rect_apt(:,:,tSteps,tAng) + [const.stim_offset(const.cond2,:),const.stim_offset(const.cond2,:)];
        
        const.probe_num_start_apt(tSteps)  =    const.bar_step_num_apt*(tSteps-1) + const.bar_step_num_apt/2 - const.probe_num/2;      % bar probe frame start
        const.probe_num_end_apt(tSteps)    =    const.bar_step_num_apt*(tSteps-1) + const.bar_step_num_apt/2 + const.probe_num/2 - 1;  % bar probe frame end
        const.resp_num_start_apt(tSteps)   =    const.probe_num_start_apt(tSteps);
        const.resp_num_end_apt(tSteps)     =    const.resp_num_start_apt(tSteps)+const.bar_step_num_apt - 1;
    end
end

% define all frames
if const.cond2 == 4
    var1 = const.bar_all_dir_run;
elseif const.cond2 == 1 || const.cond2 == 3 || const.cond2 == 2
    var1 = const.bar_hor_dir_run;
end

% define horizontal bar pass
for tAng = 1:size(var1,2)
    for nbf = 1:const.num_frame_max_hor
        
        % define bar position step number
        const.bar_steps_hor(nbf,tAng)    =   ceil(nbf/const.bar_step_num_hor);
        
        if mod(nbf,const.TR_num) == 1;          const.trial_start_hor(nbf,tAng) =   1;
        else;                                   const.trial_start_hor(nbf,tAng) =   0;
        end
        
        if mod(nbf,const.TR_num) == 0;          const.trial_end_hor(nbf,tAng)   =   1;
        else;                                   const.trial_end_hor(nbf,tAng)   =   0;
        end
        
        % define time to re-draw noise
        if mod(nbf,const.patch_num) == 1;       const.time2draw_hor(nbf,tAng)   =   1;
        else;                                   const.time2draw_hor(nbf,tAng)   =   0;
        end
        
        % define time to show probe frame
        if nbf >= const.probe_num_start_hor(const.bar_steps_hor(nbf,tAng)) && nbf <= const.probe_num_end_hor(const.bar_steps_hor(nbf,tAng)) && var1(tAng) ~= 9
            const.time2probe_hor(nbf,tAng)  =   1;
        else
            const.time2probe_hor(nbf,tAng)  =   0;
        end
        
        % define response frames
        if nbf >= const.resp_num_start_hor(const.bar_steps_hor(nbf,tAng)) && nbf <= const.resp_num_end_hor(const.bar_steps_hor(nbf,tAng)) && var1(tAng) ~= 9
            const.time2resp_hor(nbf,tAng)   =   1;
        else
            const.time2resp_hor(nbf,tAng)	=   0;
        end
        
        % define when to write the log of the probes
        if nbf == const.probe_num_start_hor(const.bar_steps_hor(nbf,tAng)) && const.time2probe_hor(nbf,tAng)
            const.time2log_hor(nbf,tAng)    =   1;
        else
            const.time2log_hor(nbf,tAng)    =   0;
        end
        
        % define response reset
        if nbf == const.resp_num_start_hor(const.bar_steps_hor(nbf,tAng))
            const.resp_reset_hor(nbf,tAng)  =   1;
        else
            const.resp_reset_hor(nbf,tAng)  =   0;
        end
    end
end

const.time2load_hor     =   [zeros(1,size(const.time2draw_hor,2));const.time2draw_hor];
const.time2load_hor     =   const.time2load_hor(1:size(const.time2draw_hor,1),:);
const.time2make_hor     =   [zeros(1,size(const.time2draw_hor,2));zeros(round(const.patch_num*0.5)-1,size(const.time2draw_hor,2));const.time2draw_hor];
const.time2make_hor     =   const.time2make_hor(1:size(const.time2draw_hor,1),:);


% define vertical bar pass
for tAng = 1:size(var1,2)
    for nbf = 1:const.num_frame_max_ver
        
        % define bar position step number
        const.bar_steps_ver(nbf,tAng)    =   ceil(nbf/const.bar_step_num_ver);
        
        if mod(nbf,const.TR_num) == 1;          const.trial_start_ver(nbf,tAng) =   1;
        else;                                   const.trial_start_ver(nbf,tAng) =   0;
        end
        
        if mod(nbf,const.TR_num) == 0;          const.trial_end_ver(nbf,tAng)   =   1;
        else;                                   const.trial_end_ver(nbf,tAng)   =   0;
        end
        
        % define time to re-draw noise
        if mod(nbf,const.patch_num) == 1;       const.time2draw_ver(nbf,tAng)   =   1;
        else;                                   const.time2draw_ver(nbf,tAng)   =   0;
        end
        
        % define time to show probe frame
        if nbf >= const.probe_num_start_ver(const.bar_steps_ver(nbf,tAng)) && nbf <= const.probe_num_end_ver(const.bar_steps_ver(nbf,tAng)) && var1(tAng) ~= 9
            const.time2probe_ver(nbf,tAng)  =   1;
        else
            const.time2probe_ver(nbf,tAng)  =   0;
        end
        
        % define response frames
        if nbf >= const.resp_num_start_ver(const.bar_steps_ver(nbf,tAng)) && nbf <= const.resp_num_end_ver(const.bar_steps_ver(nbf,tAng)) && var1(tAng) ~= 9
            const.time2resp_ver(nbf,tAng)   =   1;
        else
            const.time2resp_ver(nbf,tAng)	=   0;
        end
        
        % define when to write the log of the probes
        if nbf == const.probe_num_start_ver(const.bar_steps_ver(nbf,tAng)) && const.time2probe_ver(nbf,tAng)
            const.time2log_ver(nbf,tAng)    =   1;
        else
            const.time2log_ver(nbf,tAng)    =   0;
        end
        
        % define response reset
        if nbf == const.resp_num_start_ver(const.bar_steps_ver(nbf,tAng))
            const.resp_reset_ver(nbf,tAng)  =   1;
        else
            const.resp_reset_ver(nbf,tAng)  =   0;
        end
    end
end
const.time2load_ver     =   [zeros(1,size(const.time2draw_ver,2));const.time2draw_ver];
const.time2load_ver     =   const.time2load_ver(1:size(const.time2draw_ver,1),:);
const.time2make_ver     =   [zeros(1,size(const.time2draw_ver,2));zeros(round(const.patch_num*0.5)-1,size(const.time2draw_ver,2));const.time2draw_ver];
const.time2make_ver     =   const.time2make_ver(1:size(const.time2draw_ver,1),:);

% define aperture bar pass
for tAng = 1:size(var1,2)
    for nbf = 1:const.num_frame_max_apt
        
        % define bar position step number
        const.bar_steps_apt(nbf,tAng)    =   ceil(nbf/const.bar_step_num_apt);
        
        if mod(nbf,const.TR_num) == 1;          const.trial_start_apt(nbf,tAng) =   1;
        else;                                   const.trial_start_apt(nbf,tAng) =   0;
        end
        
        if mod(nbf,const.TR_num) == 0;          const.trial_end_apt(nbf,tAng)   =   1;
        else;                                   const.trial_end_apt(nbf,tAng)   =   0;
        end
        
        % define time to re-draw noise
        if mod(nbf,const.patch_num) == 1;       const.time2draw_apt(nbf,tAng)   =   1;
        else;                                   const.time2draw_apt(nbf,tAng)   =   0;
        end
        
        % define time to show probe frame
        if nbf >= const.probe_num_start_apt(const.bar_steps_apt(nbf,tAng)) && nbf <= const.probe_num_end_apt(const.bar_steps_apt(nbf,tAng)) && var1(tAng) ~= 9
            const.time2probe_apt(nbf,tAng)  =   1;
        else
            const.time2probe_apt(nbf,tAng)  =   0;
        end
        
        % define response frames
        if nbf >= const.resp_num_start_apt(const.bar_steps_apt(nbf,tAng)) && nbf <= const.resp_num_end_apt(const.bar_steps_apt(nbf,tAng)) && var1(tAng) ~= 9
            const.time2resp_apt(nbf,tAng)   =   1;
        else
            const.time2resp_apt(nbf,tAng)	=   0;
        end
        
        % define when to write the log of the probes
        if nbf == const.probe_num_start_apt(const.bar_steps_apt(nbf,tAng)) && const.time2probe_apt(nbf,tAng)
            const.time2log_apt(nbf,tAng)    =   1;
        else
            const.time2log_apt(nbf,tAng)    =   0;
        end
        
        % define response reset
        if nbf == const.resp_num_start_apt(const.bar_steps_apt(nbf,tAng))
            const.resp_reset_apt(nbf,tAng)  =   1;
        else
            const.resp_reset_apt(nbf,tAng)  =   0;
        end
    end
end
const.time2load_apt     =   [zeros(1,size(const.time2draw_apt,2));const.time2draw_apt];
const.time2load_apt     =   const.time2load_apt(1:size(const.time2draw_apt,1),:);
const.time2make_apt     =   [zeros(1,size(const.time2draw_apt,2));zeros(round(const.patch_num*0.5)-1,size(const.time2draw_apt,2));const.time2draw_apt];
const.time2make_apt     =   const.time2make_apt(1:size(const.time2draw_apt,1),:);

% define TR for scanner
if const.scanner
    const.TRs = 0;
    for bar_pass = 1:const.bar_dir_num        
        if var1(bar_pass) == 9
            TR_bar_pass             =   const.blk_step;
        else
            if const.cond2 == 4
                if var1(bar_pass) == 1 || var1(bar_pass) == 5
                    TR_bar_pass     =   const.bar_step_hor;
                elseif var1(bar_pass) == 3 || var1(bar_pass) == 7
                    TR_bar_pass     =   const.bar_step_ver;
                end
            else
                TR_bar_pass     =   const.bar_step_apt;
            end
        end
        const.TRs               =   const.TRs + TR_bar_pass;
    end
    fprintf(1,'\n\tScanner parameters; %1.0f TRs, %1.2f seconds, %s\n',const.TRs,const.TR_dur,datestr(seconds((const.TRs*const.TR_dur)),'MM:SS'));
end

%% Eyelink calibration value
const.maxX              =   scr.scr_sizeX*0.6;                                                 % maximum horizontal amplitude of the screen
const.maxY              =   scr.scr_sizeY*0.6;                                                 % maximum vertical amplitude of the screen
const.calib_maxX     	=   const.maxX/2;
const.calib_maxY        =   const.maxY/2;
const.calib_center      =   [scr.scr_sizeX/2,scr.scr_sizeY/2];

const.calibCoord        =   round([ const.calib_center(1),                     const.calib_center(2),...                       % 01.  center center
                                    const.calib_center(1),                     const.calib_center(2)-const.calib_maxY,...      % 02.  center up
                                    const.calib_center(1),                     const.calib_center(2)+const.calib_maxY,...      % 03.  center down
                                    const.calib_center(1)-const.calib_maxX,    const.calib_center(2),....                      % 04.  left center
                                    const.calib_center(1)+const.calib_maxX,    const.calib_center(2),...                       % 05.  right center
                                    const.calib_center(1)-const.calib_maxX,    const.calib_center(2)-const.calib_maxY,....     % 06.  left up
                                    const.calib_center(1)+const.calib_maxX,    const.calib_center(2)-const.calib_maxY,...      % 07.  right up
                                    const.calib_center(1)-const.calib_maxX,    const.calib_center(2)+const.calib_maxY,....     % 08.  left down
                                    const.calib_center(1)+const.calib_maxX,    const.calib_center(2)+const.calib_maxY,...      % 09.  right down
                                    const.calib_center(1)-const.calib_maxX/2,  const.calib_center(2)-const.calib_maxY/2,....   % 10.  mid left mid up
                                    const.calib_center(1)+const.calib_maxX/2,  const.calib_center(2)-const.calib_maxY/2,....   % 11.  mid right mid up
                                    const.calib_center(1)-const.calib_maxX/2,  const.calib_center(2)+const.calib_maxY/2,....   % 12.  mid left mid down
                                    const.calib_center(1)+const.calib_maxX/2,  const.calib_center(2)+const.calib_maxY/2]);     % 13.  mid right mid down

const.valid_maxX        =   const.calib_maxX * 0.9;
const.valid_maxY        =   const.calib_maxY * 0.9;
const.valid_center      =   const.calib_center;

const.validCoord    	=   round([ const.valid_center(1),                     const.valid_center(2),...                       % 01.  center center
                                    const.valid_center(1),                     const.valid_center(2)-const.valid_maxY,...      % 02.  center up
                                    const.valid_center(1),                     const.valid_center(2)+const.valid_maxY,...      % 03.  center down
                                    const.valid_center(1)-const.valid_maxX,    const.valid_center(2),....                      % 04.  left center
                                    const.valid_center(1)+const.valid_maxX,    const.valid_center(2),...                       % 05.  right center
                                    const.valid_center(1)-const.valid_maxX,    const.valid_center(2)-const.valid_maxY,....     % 06.  left up
                                    const.valid_center(1)+const.valid_maxX,    const.valid_center(2)-const.valid_maxY,...      % 07.  right up
                                    const.valid_center(1)-const.valid_maxX,    const.valid_center(2)+const.valid_maxY,....     % 08.  left down
                                    const.valid_center(1)+const.valid_maxX,    const.valid_center(2)+const.valid_maxY,...      % 09.  right down
                                    const.valid_center(1)-const.valid_maxX/2,  const.valid_center(2)-const.valid_maxY/2,....   % 10.  mid left mid up
                                    const.valid_center(1)+const.valid_maxX/2,  const.valid_center(2)-const.valid_maxY/2,....   % 11.  mid right mid up
                                    const.valid_center(1)-const.valid_maxX/2,  const.valid_center(2)+const.valid_maxY/2,....   % 12.  mid left mid down
                                    const.valid_center(1)+const.valid_maxX/2,  const.valid_center(2)+const.valid_maxY/2]);     % 13.  mid right mid down

end