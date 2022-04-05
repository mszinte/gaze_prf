function [noiseMatFiltNorm,vonMisses] = oriFiltNoise(fftNoiseMat,kappaPower)
    % ----------------------------------------------------------------------
    % [noiseMatFiltNorm,vonMisses] = oriFiltNoise(fftNoiseMat,kappaPower)
    % ----------------------------------------------------------------------
    % Goal of the function :
    % Filter a noise image filtered as a function of an orientation
    % width parameter (kappaPower) from a von misses distribution
    % ----------------------------------------------------------------------
    % Input(s) :
    % kappaPower = power kappa value of the von misses distribution
    %          (-1 = random, 1 = quite oriented, 2 = super oriented)
    % theta = angle of the noise
    % ----------------------------------------------------------------------
    % Output(s):
    % noiseMatFiltNorm: noise image orientation filtered and normalized
    % vonMisses: von misses filter
    % ----------------------------------------------------------------------
    % Function created by Martin SZINTE (martin.szinte@gmail.com)
    % Last update : 27 / 05 / 2018
    % Project :     --
    % Version :     1.0
    % ----------------------------------------------------------------------    
    
    %% Generate von misses distribution
    alphaVonMisses          =   linspace(0, 2*pi, size(fftNoiseMat,1))';                                                    % alpha angle of von misses
    vonMisses               =   (1/(2*pi*besseli(0,10^kappaPower))) * exp((10^kappaPower)*cos(alphaVonMisses+pi));  % von misses formula
    vonMisses               =   repmat(vonMisses,1,size(fftNoiseMat,2));                                            % repeat the matrix over the other dimension

    %% Convolve both and normalize
    noiseMatFilt            =   real(ifft2(ifftshift(fftNoiseMat.*vonMisses)));
    noiseMatFiltNorm        =   (noiseMatFilt- min(noiseMatFilt(:)))/(max(noiseMatFilt(:)) - min(noiseMatFilt(:)));
    

end
