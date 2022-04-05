% ----------------------------------------------------------------------
% noiseDemo
% ----------------------------------------------------------------------
% Goal of the function :
% Demo code of genNoise and oriFilNoise function
% see help of these function for details
% ----------------------------------------------------------------------
% Input(s) :
% none
% ----------------------------------------------------------------------
% Output(s):
% non
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 27 / 05 / 2018
% Project :     --
% Version :     1.0
% ----------------------------------------------------------------------

clear all; close all;
noiseDim = [60,80];

numCol = 3;
for row = 1:3
    % create noise
    switch row
        case 1; color{row} = 'white';
        case 2; color{row} = 'pink';
        case 3; color{row} = 'brownian';
    end
    [noiseMat{row},fftNoiseMat{row}] = genNoise(noiseDim,color{row});
    noiseMatNorm{row}  =  (noiseMat{row} - min(noiseMat{row}(:)))/(max(noiseMat{row}(:)) - min(noiseMat{row}(:)));
    
    % convolve them with different orientation filter width
    for col = 1:numCol
        kappaPower = linspace(-1,2,numCol);
        [noiseMatFiltNorm{row,col},vonMisses{row,col}] = oriFiltNoise(fftNoiseMat{row},kappaPower(col));
    end
end

%% resize image for plot
noiseImage = noiseDim*12.8;
for row = 1:3
    noiseMatNorm{row}       = imresize(noiseMatNorm{row},noiseImage);
    fftNoiseMat{row}    = imresize(fftNoiseMat{row},noiseImage);
    for col = 1:numCol
        noiseMatFiltNorm{row,col} =  imresize(noiseMatFiltNorm{row,col},noiseImage);
        vonMisses{row,col} = imresize(vonMisses{row,col},[noiseImage(1),1]);
    end
end

%% Plot results
% plot noise
figure;
set(gcf, 'Position', [0, 0, 1400, 500])
% plot convolution and filtered noise
for row = 1:3
    subplot(3,numCol*2+1,(numCol*2+1)*(row-1)+1);
    imshow(noiseMatNorm{row});
    title(sprintf('%s\nnoise',color{row}));
    
    for col = 1:numCol
        numPlot = (numCol*2+1)*(row-1)+ 1 + (col*2-1);
        subplot(3,numCol*2+1,numPlot);
        val = real(fftNoiseMat{row}.*vonMisses{row,col});
        valnorm = (val - min(val(:)))/(max(val(:)) - min(val(:)));
        imshow(valnorm);
        
        title(sprintf('noise.filter\n(k = %1.2f)',10^kappaPower(col)));
        numPlot = (numCol*2+1)*(row-1) + 2 + (col*2-1);
        subplot(3,numCol*2+1,numPlot);
        imshow(noiseMatFiltNorm{row,col});
        title(sprintf('filtered\n%s noise',color{row}));
    end
end