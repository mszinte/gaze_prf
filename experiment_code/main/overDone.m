function overDone(const,my_key)
% ----------------------------------------------------------------------
% overDone(const,my_key)
% ----------------------------------------------------------------------
% Goal of the function :
% Close screen and audio, transfer eye-link data, close files
% ----------------------------------------------------------------------
% Input(s) :
% const : struct containing constant configurations
% my_key : structure containing keyboard configurations
% ----------------------------------------------------------------------
% Output(s):
% none
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 01 / 06 / 2019
% Project :     pRF_gazeMod
% Version :     4.0
% ----------------------------------------------------------------------

% Stop recording the keyboard
% ---------------------------
for keyb = 1:size(my_key.keyboard_idx,2)
    KbQueueStop(my_key.keyboard_idx(keyb));
    KbQueueFlush(my_key.keyboard_idx(keyb));
end

% Close video file
% ----------------
if const.mkVideo
    close(const.vid_obj);
end

% Close all fid
% ------------- 
fclose(const.behav_file_fid);
if const.writeLogTxt
    fclose(const.log_file_fid);
end

% Transfer .edf file
% ------------------
if const.tracker
    statRecFile = Eyelink('ReceiveFile',const.eyelink_temp_file,const.eyelink_temp_file);
    
    if statRecFile ~= 0
        fprintf(1,'\n\n\tEyelink EDF file correctly transfered\n');
    else
        fprintf(1,'\n\n\tError in Eyelink EDF file transfer\n');
        statRecFile2 = Eyelink('ReceiveFile',const.eyelink_temp_file,const.eyelink_temp_file);
        if statRecFile2 == 0
            fprintf(1,'\n\n\tEyelink EDF file is now correctly transfered\n');
        else
            fprintf(1,'\n\n\t!!!!! Error in Eyelink EDF file transfer !!!!!\n');
        end
    end
end

% Close link with eye tracker
% ---------------------------
if const.tracker 
    Eyelink('CloseFile');
    WaitSecs(2.0);
    Eyelink('Shutdown');
    WaitSecs(2.0); 
end

% Rename eye tracker file
% -----------------------
if const.tracker 
    oldDir = const.eyelink_temp_file;
    newDir = const.eyelink_local_file;
    movefile(oldDir,newDir);
end

% Close Psychtoolbox screen
% -------------------------
ListenChar(1);ShowCursor;
Screen('CloseAll');

% Print block duration
% --------------------
timeDur=toc/60;
fprintf(1,'\n\tThis part of the experiment took : %2.0f min.\n\n',timeDur);

end