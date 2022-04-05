function behav_results(subject,num_run,eyetracker)
% ----------------------------------------------------------------------
% behav_results(subject,num_run,eyetracker)
% ----------------------------------------------------------------------
% Goal of the function :
% Compute and plot behavioral results
% ----------------------------------------------------------------------
% Input(s) :
% subject : subject name
% num_run : number of runs to analyse
% eyetracker : eye tracking data
% ----------------------------------------------------------------------
% Output(s):
% => pdf figures /data/sub-XXX/add/sub-XXX_behav_results.pdf
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 13 / 09 / 2019
% Project :     pRFseqTest
% Version :     1.0
% ----------------------------------------------------------------------

%% Get data
% Get behavioral data
fprintf(1,'\n\tProcessing data...\n');

file_dir = sprintf('%s/data/%s',cd,subject);

task1_txt = 'AttendStim';
task2_1_txt = '_acq-2p5mm';
task2_2_txt = '_acq-2mm';

list_filename = {   sprintf('%s_task-%s%s_run-01',subject,task1_txt,task2_1_txt),...
                    sprintf('%s_task-%s%s_run-01',subject,task1_txt,task2_2_txt),...
                    sprintf('%s_task-%s%s_run-02',subject,task1_txt,task2_1_txt),...
                    sprintf('%s_task-%s%s_run-02',subject,task1_txt,task2_2_txt),...
                    sprintf('%s_task-%s%s_run-03',subject,task1_txt,task2_1_txt),...
                    sprintf('%s_task-%s%s_run-03',subject,task1_txt,task2_2_txt),...
                    sprintf('%s_task-%s%s_run-04',subject,task1_txt,task2_1_txt),...
                    sprintf('%s_task-%s%s_run-04',subject,task1_txt,task2_2_txt),...
                    sprintf('%s_task-%s%s_run-05',subject,task1_txt,task2_1_txt),...
                    sprintf('%s_task-%s%s_run-05',subject,task1_txt,task2_2_txt)};

% Get behavioral data
% -------------------
time_last_run = 0;
response_vals = [];
stim_stair_vals = [];
bar_directions = [];
time_runs = [];

for t_run = 1:num_run
    
    % extract start time
    tsv_filename = sprintf('%s/func/%s_events.tsv',file_dir,list_filename{t_run});
    val = tdfread(tsv_filename);
    
    % transform to matlab nan
    % response value
    response_val = zeros(size(val.response_val,1),1);
    for t_line = 1:size(val.response_val,1)
        if strcmp(val.response_val(t_line,:),'n/a');        response_val(t_line,1) = nan;
        else                                                response_val(t_line,1) = str2double(val.response_val(t_line,1));
        end
    end
    val.response_val = response_val;
    
    % concatenate values
    response_vals = [response_vals;val.response_val];
    stim_stair_vals = [stim_stair_vals;val.stim_stair_val];
    bar_directions = [bar_directions;val.bar_direction];
    
    % define time
    time_start(t_run) = val.onset(1,1);
    time_end(t_run) = val.onset(end,1);
    time_run = val.onset-time_start(t_run)+time_last_run;
    time_last_run = max(time_run);
    time_runs = [time_runs;time_run];
    
end
time_prc = time_runs/sum(time_end - time_start);

performances = [];
for t = 1:size(response_vals,1)
    performance = nanmean(response_vals(1:t));
    performances = [performances;performance];
end

% Get eye data
% ------------
if ismac
    edf2asc_dir = '/Applications/Eyelink/EDF_Access_API/Example';
    end_file = '';
elseif ispc 
    edf2asc_dir = 'C:/Users/invibe/Experiments/pRFseqTest/stats/';
    end_file ='.exe';
end

if eyetracker
    time_last_run_eye   =   0;
    fix_accuracy        =   [];
    eye_data_runs       =   [];
    
    for t_run = 1:num_run
        
        mat_filename = sprintf('%s/add/%s_matFile.mat',file_dir,list_filename{t_run});
        load(mat_filename);
        edf_filename = sprintf('%s/func/%s_eyeData',file_dir,list_filename{t_run});
        
        % get .msg and .dat file
        if ~exist(sprintf('%s.dat',edf_filename),'file') || ~exist(sprintf('%s.msg',edf_filename),'file')
            [~,~] = system(sprintf('%s/edf2asc%s %s.edf -e -y',edf2asc_dir,end_file,edf_filename));
            movefile(sprintf('%s.asc',edf_filename),sprintf('%s.msg',edf_filename));
            [~,~] = system(sprintf('%s/edf2asc%s %s.edf -s -miss -1.0 -y',edf2asc_dir,end_file,edf_filename));
            movefile(sprintf('%s.asc',edf_filename),sprintf('%s.dat',edf_filename));
        end
        
        % get first and last time
        msgfid = fopen(sprintf('%s.msg',edf_filename),'r');
        first_last_time = 0;
        first_time = 0;
        last_time = 0;
        mri_trigger_eye = [];
        while ~first_last_time
            line_read = fgetl(msgfid);
            if ~isempty(line_read)                           % skip empty lines
                la = textscan(line_read,'%s');
                % get first time
                if size(la{1},1) > 6
                    if strcmp(la{1}(3),'bar') && strcmp(la{1}(4),'pass') && strcmp(la{1}(5),'1') && strcmp(la{1}(6),'started') && ~first_time
                        time_start_eye(t_run) = str2double(la{1}(2));
                        first_time = 1;
                    end
                    
                    if strcmp(la{1}(3),'bar') && strcmp(la{1}(4),'pass') && strcmp(la{1}(5),'9') && strcmp(la{1}(6),'stopped') && ~last_time
                        time_end_eye(t_run) = str2double(la{1}(2));
                        last_time = 1;
                    end
                    
                    if strcmp(la{1}(7),'mri_trigger')
                        mri_trigger_eye = [mri_trigger_eye;str2double(la{1}(2))];
                    end
                    
                end
                
                if first_time && last_time
                    first_last_time = 1;
                    fclose(msgfid);
                end
            end
        end
        
        % load eye coord data
        datafid = fopen(sprintf('%s.dat',edf_filename),'r');
        eye_dat = textscan(datafid,'%f%f%f%f%s');
        fclose(datafid);
        eye_data_run = [eye_dat{1},eye_dat{2},eye_dat{3}];
        eye_data_run = eye_data_run(eye_data_run(:,1)>=time_start_eye(t_run) & eye_data_run(:,1)<=time_end_eye(t_run),:);
        % col 1 = time
        % col 2 = eye x coord
        % col 3 = eye y coord
        
        % compute time relative to start of trial and across blocks
        eye_data_run(:,1) =  (eye_data_run(:,1)-time_start_eye(t_run))+time_last_run_eye;
        time_last_run_eye = eye_data_run(end,1);
        
        eye_data_runs = [eye_data_runs;eye_data_run];
        mri_trigger_eye_runs{t_run} = mri_trigger_eye;
    end
    
    % delete blink time
    blinkNum = 0;
    blink_start = 0;
    for tTime = 1:size(eye_data_runs,1)
        if ~blink_start
            if eye_data_runs(tTime,2)==-1
                blinkNum = blinkNum + 1;
                timeBlinkOnset = eye_data_runs(tTime,1);
                blink_start = 1;
                blink_onset_offset(blinkNum,:) = [timeBlinkOnset,NaN];
            end
        end
        if blink_start
            if eye_data_runs(tTime,2)~=-1
                timeBlinkOfffset = eye_data_runs(tTime,1);
                blink_start = 0;
                blink_onset_offset(blinkNum,2) = timeBlinkOfffset;
            end
        end
    end
    
    % nan record around detected blinks
    befDurBlink = 300;   % duration before blink
    aftDurBlink = 300;   % duration after blink
    eye_data_runs_no_blink = eye_data_runs;
    for tBlink = 1:blinkNum
        blink_onset_offset(tBlink,:) = [blink_onset_offset(tBlink,1)-befDurBlink,blink_onset_offset(tBlink,1)+aftDurBlink];
        eye_data_runs_no_blink(eye_data_runs(:,1) >= blink_onset_offset(tBlink,1) & eye_data_runs_no_blink(:,1) <= blink_onset_offset(tBlink,2),2) = NaN;
        eye_data_runs_no_blink(eye_data_runs(:,1) >= blink_onset_offset(tBlink,1) & eye_data_runs_no_blink(:,1) <= blink_onset_offset(tBlink,2),3) = NaN;
    end
    
    % compute in time percentage between start and end
    time_prc_eye = eye_data_runs(:,1)/sum(time_end_eye - time_start_eye);
    
    % put eye coordinates in deg from center (flip y axis)
    screen_size = [config.scr.scr_sizeX,config.scr.scr_sizeY];
    ppd = config.const.ppd;
    eye_data_runs(:,2) = (eye_data_runs(:,2) - (screen_size(1)/2))/ppd;
    eye_data_runs(:,3) = (-1*(eye_data_runs(:,3) - screen_size(2)/2))/ppd;
    
    eye_data_runs_no_blink(:,2) = (eye_data_runs_no_blink(:,2) - (screen_size(1)/2))/ppd;
    eye_data_runs_no_blink(:,3) = (-1*(eye_data_runs_no_blink(:,3) - screen_size(2)/2))/ppd;
       
    % compute mean accuracy and precision
    fix_pos = [0,0];
    fix_accuracy = [fix_accuracy;sqrt((eye_data_runs_no_blink(:,2) - fix_pos(1)).^2 + (eye_data_runs_no_blink(:,3) - fix_pos(2)).^2)];
    
    % compute mean fixation accuracy
    accuracy_val = nanmean(fix_accuracy);
    
    % compute mean fixation precision
    precision_val = nanstd(fix_accuracy,1);
    
    % compute fixation heatmap
%     eye_data_runs_resample  = resample(eye_data_runs_no_blink,1,100);
%     time_prc_eye_resample  = resample(time_prc_eye,1,100);
    eye_data_runs_no_blink_movmean = eye_data_runs_no_blink;
    eye_data_runs_no_blink_movmean (:,2) = movmean(eye_data_runs_no_blink_movmean (:,2),5000,'omitnan');
    eye_data_runs_no_blink_movmean (:,3) = movmean(eye_data_runs_no_blink_movmean (:,3),5000,'omitnan');
    
    radMaxX     = ceil(config.scr.x_mid/ppd);
    radMaxY     = ceil(config.scr.y_mid/ppd);
    [~,densityFix,xHeatMap,yHeatMap]=kde2d([eye_data_runs(:,2),eye_data_runs(:,3)],[-radMaxX,-radMaxY],[radMaxX,radMaxY],2^5);
    densityFix_min = min(min(densityFix));
    densityFix_max = max(max(densityFix));
    densityFix = (densityFix - densityFix_min)./(densityFix_max-densityFix_min);
    
    % compute fixation heatmap zoom
    radMaxX_zoom     = ceil(config.scr.x_mid/ppd)/3;
    radMaxY_zoom     = ceil(config.scr.y_mid/ppd)/3;
    [~,densityFix_zoom,xHeatMap_zoom,yHeatMap_zoom]=kde2d([eye_data_runs(:,2),eye_data_runs(:,3)],[-radMaxX_zoom,-radMaxY_zoom],[radMaxX_zoom,radMaxY_zoom],2^5);
    densityFix_min_zoom = min(min(densityFix_zoom));
    densityFix_max_zoom = max(max(densityFix_zoom));
    densityFix_zoom = (densityFix_zoom - densityFix_min_zoom)./(densityFix_max_zoom-densityFix_min_zoom);
    
    
    
end

%% Plot figure
fprintf(1,'\n\tPlotting figure...\n');
% set figure
close all

numRow      = 2;
if eyetracker  
    numRow      =   2;
    numColumn   =   3;
else
    numRow      =   1;
    numColumn   =   2;
end
white                   = [1,1,1];
black                   = [0,0,0];
gray                    = [0.6,0.6,0.6];
beige                   = [230,230,230]/255;
sizeX                   = 1920/4;
sizeY                   = 1080/4;
figSize_X               = sizeX*numColumn;
figSize_Y               = sizeY*numRow;
start_X                 = 0;
start_Y                 = 0;
paperSize               = [figSize_X/30,figSize_Y/30];
paperPos                = [0 0 paperSize(1) paperSize(2)];
name                    = sprintf('%s results',subject);
f                       = figure;
set(f,'Name',name,'PaperUnits','centimeters','PaperPosition',paperPos,'Color',[1 1 1],'PaperSize',paperSize);
set(f,'Position',[start_X,start_Y,figSize_X,figSize_Y]);
mapVal                  = viridis(100);
resCol                  = black;
colormap(mapVal);


% plot data
xyscale     = sizeY/sizeX;
mergin      = 0.05;
tickratio   = 0.03;

for tRow = 1:numRow
    
    
    for tCol = 1:numColumn
       
        x_start = (tCol-1)/numColumn;
        y_start = (numRow-tRow)/numRow;
        x_size = 1/numColumn;
        y_size = 1/numRow;
        axes('position',[x_start,y_start,x_size,y_size]);
        
        hold on
        
        if tRow == 1
            if tCol == 1
                xPlot       = time_prc;
                yPlot       = stim_stair_vals;
                xlim        = [0,1];
                ylim        = [0,15];
                xtick       = 0:0.1:1;
                ytick       = 0:5:15;
                xlabel      = 'Time (%)';
                ylabel      = 'Staircase value (a.u.)';
                fig_title   = 'Staircase';
            elseif tCol == 2
                xPlot       = time_prc;
                yPlot       = performances;
                xlim        = [0,1];
                ylim        = [0,1];
                xtick       = 0:0.1:1;
                ytick       = 0:0.25:1;
                xlabel      = 'Time (%)';
                ylabel      = 'Performance (%)';
                fig_title   = 'Behavioral performance';
            elseif tCol == 3
                xHeatMap_val    = xHeatMap;
                yHeatMap_val    = yHeatMap;
                densityFix_val  = densityFix;
                xlim        = [-radMaxX,radMaxX];
                ylim        = [-radMaxY,radMaxY];
                xtick       = linspace(xlim(1),xlim(2),15);
                ytick       = linspace(ylim(1),ylim(2),7);
                xlabel      = 'Horizontal coord. (dva)';
                ylabel      = 'Vertical coord. (dva)';
                fig_title   = 'Fixation heatmap (zoom)';
            end
        elseif tRow == 2
            if tCol == 1
                xPlot       = time_prc_eye;
                yPlot       = eye_data_runs_no_blink_movmean(:,2);
                yPlot2      = eye_data_runs(:,2);
                xlim        = [0,1];
                ylim        = [-5,5];
                xtick       = 0:0.1:1;
                ytick       = linspace(ylim(1),ylim(2),11);
                ylabel      = 'Eye horizontal position (dva)';
                fig_title   = 'Eye horizontal position';
            elseif tCol == 2
                xPlot       = time_prc_eye;
                yPlot       = eye_data_runs_no_blink_movmean(:,3);
                yPlot2      = eye_data_runs_no_blink(:,3);
                xlim        = [0,1];
                ylim        = [-5,5];
                xtick       = 0:0.1:1;
                ytick       = linspace(ylim(1),ylim(2),11);
                ylabel      = 'Eye vertical position (dva)';
                fig_title   = 'Eye vertical position';
            elseif tCol == 3
                xHeatMap_val    = xHeatMap_zoom;
                yHeatMap_val    = yHeatMap_zoom;
                densityFix_val  = densityFix_zoom;
                xlim        = [-radMaxX_zoom,radMaxX_zoom];
                ylim        = [-radMaxY_zoom,radMaxY_zoom];
                xtick       = linspace(xlim(1),xlim(2),15);
                ytick       = linspace(ylim(1),ylim(2),7);
                xlabel      = 'Horizontal coord. (dva)';
                ylabel      = 'Vertical coord. (dva)';
                fig_title   = 'Fixation heatmap (zoom)';
            end
        end
        yrange      = ylim(2)-ylim(1);
        xrange      = xlim(2)-xlim(1);
        legSize     = 0.05*xrange;
        
        % plot back figures
        patch([xlim(1),xlim(2),xlim(2),xlim(1)],[ylim(1),ylim(1),ylim(2),ylim(2)],beige,'linestyle','none');
        
        % plot data
        if tCol < 3
            
            if tRow == 2
                plot(xPlot,yPlot2,'LineWidth',2,'Color',[0.95,0.95,0.95]);
            end
            plot(xPlot,yPlot,'LineWidth',2,'Color',resCol);
            
        else
            contourf(xHeatMap_val,yHeatMap_val,densityFix_val,10,'linestyle','none');
        end
        
        % plot white hiders
        patch([xlim(1),xlim(2),xlim(2),xlim(1)],[ylim(1),ylim(1),ylim(1)-yrange*2,ylim(1)-yrange*2],white,'linestyle','none');
        patch([xlim(1),xlim(2),xlim(2),xlim(1)],[ylim(2),ylim(2),ylim(2)+yrange*2,ylim(2)+yrange*2],white,'linestyle','none');
        
        % plot fixation position
        if tCol < 3 && tRow == 2
            xFix = xlim(2)+(xrange*mergin);
            plot(xFix,0,'<','MarkerFaceColor',black,'MarkerEdgeColor','none','MarkerSize',8);
            plot(xFix,0,'<','MarkerFaceColor',black,'MarkerEdgeColor','none','MarkerSize',8);
        end
        if tCol > 2
            plot(linspace(xlim(1),xlim(2),30),linspace(xlim(1),xlim(2),30)*0,'Color',white,'LineWidth',0.5);
            plot(linspace(ylim(1),ylim(2),30)*0,linspace(ylim(1),ylim(2),30),'Color',white,'LineWidth',0.5);            
        end
        
        % plot colormap
        % colorbar
        if tCol > 2
            yPlotCB = linspace(ylim(1),ylim(2),10);
            xPlotCB = yPlotCB*0 + xlim(2)+(xrange*mergin*3);
            ytickCB = linspace(ylim(1),ylim(2),5);
            ytickValCB = linspace(0,1,5);
            xPlotCBTickTxt =  xlim(2)+(xrange*mergin*5);
            xPlotCBTitleTxt =  xlim(2)+(xrange*mergin*7);
            plot(xPlotCB,yPlotCB,'Color',black,'LineWidth',1);
            for tickCB = 1:size(ytickCB,2)
                xPlotCBTick = linspace(xlim(2)+(xrange*mergin*3),xlim(2)+(xrange*mergin*3)+(xrange*tickratio*xyscale),10);
                yPlotCBTick = xPlotCBTick*0+ytickCB(tickCB);
                plot(xPlotCBTick,yPlotCBTick,'Color',black,'LineWidth',1);
                text(xPlotCBTickTxt,yPlotCBTick(1),sprintf('%1.2g',ytickValCB(tickCB)),'Hor','center','Ver','Middle')
            end
            text(xPlotCBTitleTxt,mean(yPlotCB),'Normalized density (%)','Hor','center','Ver','Middle','Rotation',90)
            
            xGridCB = [xlim(2)+(xrange*mergin),xlim(2)+(xrange*mergin*2)];
            yGridCB = linspace(ylim(1),ylim(2),100);
            [colorPosXCB,colorPosYCB,colorPosZCB] = meshgrid(xGridCB,yGridCB,0);
            m1 = surf(colorPosXCB,colorPosYCB,colorPosZCB,[linspace(ytickValCB(1),ytickValCB(end),100)',linspace(ytickValCB(1),ytickValCB(end),100)']);
            set(m1,'LineStyle','none','FaceColor','flat')
        end
    
        % plot xaxis/xaxis tick
        xPlotxAxis = linspace(xlim(1),xlim(2),30);
        yPlotxAxis = xPlotxAxis*0 + ylim(1)-(yrange*mergin*2);
        yPlotxAxisTickTxt =  ylim(1)-(yrange*mergin*4);
        yPlotxAxisTitleTxt =  ylim(1)-(yrange*mergin*6);
        plot(xPlotxAxis,yPlotxAxis,'Color',black,'LineWidth',1);
        for tickXaxis = 1:size(xtick,2)
            yPlotxAxisTick = linspace(ylim(1)-(yrange*mergin*2),ylim(1)-(yrange*mergin*2)-(yrange*tickratio),10);
            xPlotxAxisTick = yPlotxAxisTick*0+xtick(tickXaxis);
            plot(xPlotxAxisTick,yPlotxAxisTick,'Color',black,'LineWidth',1);
            text(xPlotxAxisTick(1),yPlotxAxisTickTxt,sprintf('%1.2g',xtick(tickXaxis)),'Hor','center','Ver','Middle')
        end
        text(mean(xPlotxAxis),yPlotxAxisTitleTxt,xlabel,'Hor','center','Ver','Middle')
        
        % plot yaxis/yaxis tick
        yPlotyAxis = linspace(ylim(1),ylim(2),30);
        xPlotyAxis = yPlotxAxis*0 + xlim(1)-(xrange*mergin);
        xPlotyAxisTickTxt =  xlim(1)-(xrange*mergin*3);
        xPlotyAxisTitleTxt =  xlim(1)-(xrange*mergin*5);
        plot(xPlotyAxis,yPlotyAxis,'Color',black,'LineWidth',1);
        for tickYaxis = 1:size(ytick,2)
            xPlotyAxisTick = linspace(xlim(1)-(xrange*mergin),xlim(1)-(xrange*mergin)-(xrange*tickratio*xyscale),10);
            yPlotyAxisTick = xPlotxAxisTick*0+ytick(tickYaxis);
            plot(xPlotyAxisTick,yPlotyAxisTick,'Color',black,'LineWidth',1);
            if tRow > 2 && tCol == 2
                text(xPlotyAxisTickTxt,yPlotyAxisTick(1),sprintf('%1.0f',ytick(tickYaxis)),'Hor','center','Ver','Middle')
            else
                text(xPlotyAxisTickTxt,yPlotyAxisTick(1),sprintf('%1.2g',ytick(tickYaxis)),'Hor','center','Ver','Middle')
            end
        end
        text(xPlotyAxisTitleTxt,mean(yPlotyAxis),ylabel,'Hor','center','Ver','Middle','Rotation',90)
        
        % plot title
        xPlotTitle =  xPlotxAxis(1);
        yPlotTitle =  ylim(2)+(yrange*mergin*3);
        text(xPlotTitle,yPlotTitle,fig_title,'Hor','left','Ver','middle','Color',black,'FontWeight','bold')
        
        % plot run lines
        if tCol < 3
            yRun = linspace(ylim(1),ylim(2),10);
            for t_run_line = 1:num_run
                xRunStart = (t_run_line-1)/num_run;
                xRunEnd   = t_run_line/num_run;
                xRunTxt   = (xRunStart+xRunEnd)/2;
                plot(yRun*0+xRunStart,yRun,'Color',gray,'LineWidth',1);
                plot(yRun*0+xRunEnd,yRun,'Color',gray,'LineWidth',1);
                text(xRunTxt,ylim(1)+(yrange*mergin),sprintf('Run %i',t_run_line),'Hor','center','Ver','Middle','Color',gray,'FontSize',8)
            end
        end
        
        % plot threshold
        if tRow == 1 && tCol == 2
            xStairTh = linspace(xlim(1),xlim(2),30);
            threVal = 0.794;
            yStairTh = xStairTh*0 + threVal;
            plot(xStairTh,yStairTh,'-','Color',black,'LineWidth',1);
            text(xStairTh(end),yStairTh(1)-0.05,sprintf('Threshold = %1.2f   ',threVal),'Hor','right','Ver','Middle','Color',black,'FontSize',8)
        end
        
        % plot legend
        if tCol == 3
            xPlotAcc = xlim(2);
            yPlotAcc = ylim(2)+(yrange*mergin*(4));
            text(xPlotAcc+xrange*0.02,yPlotAcc,sprintf('Fixation accuracy = %1.2f dva',accuracy_val),'Hor','right','Ver','Middle','FontSize',7)
            xPlotPre = xlim(2);
            yPlotPre = ylim(2)+(yrange*mergin*(2));
            text(xPlotPre+xrange*0.02,yPlotPre,sprintf('Fixation precision = %1.2f dva',precision_val),'Hor','right','Ver','Middle','FontSize',7)
        elseif tRow == 2 && tCol < 3
            xPlotBlink = xlim(2);
            yPlotBlink = ylim(2)+(yrange*mergin*(4));
            text(xPlotBlink+xrange*0.02,yPlotBlink,sprintf('Blinks = %i',blinkNum),'Hor','right','Ver','Middle','FontSize',7)

        end
        
        % plot block time
        if tCol < 3
            xPlotTask = time_prc;
            xPlotTask(bar_directions==9) = NaN;
            yPlotTask = xPlotTask*0 + ylim(1)-(yrange*mergin);
            plot(xPlotTask,yPlotTask,'LineWidth',5,'Color',resCol);
        end
        
        % set figure 
        set(gca,'XLim',xlim+[-xrange*0.35,xrange*0.45],'YLim',ylim+[-yrange*0.4,yrange*0.4],'XTicklabel','','YTicklabel','','XColor',white,'YColor',white)
        
    end
end

% save figure
saveas(f,sprintf('%s/add/%s_behav_results.pdf',file_dir,subject))

end