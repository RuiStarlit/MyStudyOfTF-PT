% =========================================================================
%   将给定的DNA序列转化为在K-mer为3下的数字信号，然后在其中添加噪声
%   最后通过隐式马尔可夫模型把添加过噪声的DNA还原
% =========================================================================

SNR = 30;   % SIGNAL NOISE RATIO（信噪比）

% 给定DNA序列（存储在文件中）
fo = fopen('dnaSequenceSample.txt', 'r');
dnaSeq = fgets(fo);
fclose(fo);

K = 3;  % K-mer为3

% 将DNA序列转化为数字信号
digitSignal = dnaSeq2Digit(dnaSeq);

% 在数字信号中添加高斯噪声
digitSignalNoise = noiseAdd(digitSignal, SNR);
% 可在noiseAdd函数中设置随机数生成器状态（每次噪声一样还是不同）
% =========================================================================

% 在已知HMM模型的参数下，可以用Viterbi算法求解观察序列的
% 最佳隐状态序列
dSRecovery = viterbiDecoder(digitSignalNoise);

% 将数字信号恢复成DNA序列
dnaSeqRecovery = digit2dnaSeq(dSRecovery);

% 将原DNA序列和HMM模型预测的序列进行对比
fprintf('原DNA序列为：\n');
disp(dnaSeq);
fprintf('基于HMM模型预测的DNA序列为：\n');
disp(dnaSeqRecovery);
err=sum(dnaSeq ~= dnaSeqRecovery);
fprintf('预测结果中碱基错误数量为：%d ,错误率为：%4.2f%% \n',err,err/length(dnaSeq)*100);
