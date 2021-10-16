function y = noiseAdd(x, snr)
%

% rng('default'); %将随机数生成器设为默认，这样每次的噪声将相同。可以注释本行
L = length(x);
SNR = 10^(snr/10); % 将SNR从dB恢复为线性
Esym = sum(abs(x).^2)/(L); % 能量的谱密度（Energy Spectrum Density，ESD）
N0 = Esym/SNR; % 噪声能量的谱密度

noiseSigma = sqrt(N0); % 标准加性高斯白噪声(Additive White Gaussian Noise)
n = noiseSigma * randn(1, L);

y = x + n; % 实际信号（添噪后）