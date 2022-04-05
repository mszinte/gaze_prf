function [noiseMatFiltNorm] = genNoisePatch(const,kappa)
% ----------------------------------------------------------------------
% [noiseMatFiltNorm] = genNoisePatch(const,kappa)
% ----------------------------------------------------------------------
% Goal of the function :
% Create noise patches
% ----------------------------------------------------------------------
% Input(s) :
% const : struct containing constant configurations
% kappa : dispersion parameter of the von misses filter
% ----------------------------------------------------------------------
% Output(s):
% noiseMatFiltNorm: patch
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 31 / 05 / 2018
% Project :     pRF_gazeMod
% Version :     3.0
% ----------------------------------------------------------------------

% define original noise patch size
noiseDim                =  const.native_noise_dim;                                                                  % starting size of the patch

% create pink noise stimuli
if strcmp(const.noise_color,'pink')
    colVal                  =   1;                                                                                  % pink 1/f noise
elseif strcmp(const.noise_color,'brownian')
    colVal                  =   2;                                                                                  % brownian 2/f noise
elseif strcmp(const.noise_color,'white')
    colVal                  =   0;                                                                                  % white 0/f nosie
end

freqD1                  =   repmat([(0:floor(noiseDim(1)/2)) -(ceil(noiseDim(1)/2)-1:-1:1)]'/...
                                                                        noiseDim(1),1,noiseDim(2));                 % set of frequencies along the first dimension
freqD2                  =   repmat([(0:floor(noiseDim(2)/2)) -(ceil(noiseDim(2)/2)-1:-1:1)]/...
                                                                        noiseDim(2),noiseDim(1),1);                 % set of frequencies along the second dimension.
pSpect                  =   (freqD1.^2 + freqD2.^2).^(-colVal/2);                                                   % power spectrum
pSpect(pSpect==inf)     =   0;                                                                                      % set any infinities to zero
phi                     =   rand(noiseDim);                                                                         % generate a grid of random phase shifts
noiseMat                =   real(ifft2(pSpect.^0.5 .* (cos(2*pi*phi)+1i*sin(2*pi*phi))));                           % real component of inverse Fourier transform to obtain the the spatial pattern
fftNoiseMat             =   fftshift(fft2(noiseMat-mean(noiseMat(:)), noiseDim(1), noiseDim(2)));                   % fast fourier transform

% Generate von misses distribution
alphaVonMisses          =   linspace(0, 2*pi, size(fftNoiseMat,1))';                                                % alpha angle of von misses
vonMisses               =   (1/(2*pi*besseli(0,kappa))) * exp((kappa)*cos(alphaVonMisses+pi));                      % von misses formula
vonMisses               =   repmat(vonMisses,1,size(fftNoiseMat,2));                                                % repeat the matrix over the other dimension

% Convolve both and normalize
noiseMatFilt            =   real(ifft2(ifftshift(fftNoiseMat.*vonMisses)));                                         % convolution
noiseMatFiltNorm        =   (noiseMatFilt- min(noiseMatFilt(:)))/(max(noiseMatFilt(:)) - min(noiseMatFilt(:)));     % normalization


end