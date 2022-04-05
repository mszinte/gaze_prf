function [noiseMat,fftNoiseMat] = genNoise(noiseDim,color)
    % ----------------------------------------------------------------------
    % [noiseMat,fftNoiseMat] = genNoise(dimension,color)
    % ----------------------------------------------------------------------
    % Goal of the function :
    % Create an noise image of different color
    % ----------------------------------------------------------------------
    % Input(s) :
    % noiseDim = noise patch dimension in pixels ([100,100])
    % color = color of the noise ('pink','brown','white')
    % ----------------------------------------------------------------------
    % Output(s):
    % noiseMat: noise image not normalized
    % fftNoiseMat: fast fourier transform of the noise
    % ----------------------------------------------------------------------
    % Function created by Martin SZINTE (martin.szinte@gmail.com)
    % Last update : 27 / 05 / 2018
    % Project :     --
    % Version :     1.0
    % ----------------------------------------------------------------------
    
    if strcmp(color,'pink')
        colVal = 1;
    elseif strcmp(color,'brownian')
        colVal = 2;
    elseif strcmp(color,'white')
        colVal = 0;
    end

    %% Generate noise
    freqD1                  =   repmat([(0:floor(noiseDim(1)/2)) -(ceil(noiseDim(1)/2)-1:-1:1)]'/...
                                                                            noiseDim(1),1,noiseDim(2));       % set of frequencies along the first dimension
    freqD2                  =   repmat([(0:floor(noiseDim(2)/2)) -(ceil(noiseDim(2)/2)-1:-1:1)]/...
                                                                            noiseDim(2),noiseDim(1),1);       % set of frequencies along the second dimension. 
    pSpect                  =   (freqD1.^2 + freqD2.^2).^(-colVal/2);                                         % power spectrum
    pSpect(pSpect==inf)     =   0;                                                                            % set any infinities to zero 
    phi                     =   rand(noiseDim);                                                               % generate a grid of random phase shifts
    noiseMat                =   real(ifft2(pSpect.^0.5 .* (cos(2*pi*phi)+1i*sin(2*pi*phi))));                 % real component of inverse Fourier transform to obtain the the spatial pattern
    fftNoiseMat             =   fftshift(fft2(noiseMat-mean(noiseMat(:)), noiseDim(1), noiseDim(2)));         % fast fourier transform

end
