function [my_key]=keyConfig
% ----------------------------------------------------------------------
% [my_key]=keyConfig
% ----------------------------------------------------------------------
% Goal of the function :
% Unify key names and define structure containing each key names
% ----------------------------------------------------------------------
% Input(s) :
% none
% ----------------------------------------------------------------------
% Output(s):
% my_key : structure containing keyboard configurations
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 01 / 06 / 2019
% Project :     pRF_gazeMod
% Version :     4.0
% ----------------------------------------------------------------------

KbName('UnifyKeyNames');

my_key.trVal            =   't';                    % TR
my_key.left1Val         =   'd';                    % left button box button 1 (Leftmost->Rightmost)
my_key.left2Val         =   'n';                    % left button box button 2 (Leftmost->Rightmost)
my_key.left3Val         =   'w';                    % left button box button 3 (Leftmost->Rightmost)
my_key.left4Val         =   'e';                    % left button box button 4 (Leftmost->Rightmost)
my_key.right1Val        =   'b';                    % right button box button 1 (Leftmost->Rightmost)
my_key.right2Val        =   'y';                    % right button box button 2 (Leftmost->Rightmost)
my_key.right3Val        =   'g';                    % right button box button 3 (Leftmost->Rightmost)
my_key.right4Val        =   'r';                    % right button box button 4 (Leftmost->Rightmost)
my_key.escapeVal        =   'escape';               % escape button
my_key.spaceVal         =   'space';                % space button

my_key.escape           =   KbName(my_key.escapeVal);
my_key.space            =   KbName(my_key.spaceVal);
my_key.tr               =   KbName(my_key.trVal);
my_key.left1            =   KbName(my_key.left1Val);
my_key.left2            =   KbName(my_key.left2Val);
my_key.left3            =   KbName(my_key.left3Val);
my_key.left4            =   KbName(my_key.left4Val);
my_key.right1           =   KbName(my_key.right1Val);
my_key.right2           =   KbName(my_key.right2Val);
my_key.right3           =   KbName(my_key.right3Val);
my_key.right4           =   KbName(my_key.right4Val);


my_key.keyboard_idx     =   GetKeyboardIndices;
for keyb = 1:size(my_key.keyboard_idx,2)
    KbQueueCreate(my_key.keyboard_idx(keyb));
    KbQueueFlush(my_key.keyboard_idx(keyb));
    KbQueueStart(my_key.keyboard_idx(keyb));
end

[~,keyCodeMat]   = KbQueueCheck(my_key.keyboard_idx(1));
my_key.keyCodeNum  = numel(keyCodeMat,1);

end