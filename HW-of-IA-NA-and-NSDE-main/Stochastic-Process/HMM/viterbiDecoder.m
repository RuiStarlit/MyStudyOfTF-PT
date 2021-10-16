function dSRecovery = viterbiDecoder(dSN)
% 
dnaSeqLen = length(dSN);
% 数字信号中所有可能的状态:1 to 64
maxTri = 64;
tri = 1 : 1 : maxTri;

% 转移概率矩阵A
A = 0.25*ones(4,4);

% Emission probability B
%sigma = var(dSN); 
sigma = 1; %最简单的情况
b = 1./( sqrt(2*pi*sigma) * exp(((dSN(1) - tri).^2))/sigma);
% Previous forward path probability from the previous time step
a = ones(1,dnaSeqLen);
x = b;
% Initialize
dSRecovery = zeros(1, dnaSeqLen);

for k = 2 : dnaSeqLen
    % 跑遍每种情况
    b = 1./( sqrt(2*pi*sigma) * exp(((dSN(k) - tri).^2))/sigma);
    for i = 1 : maxTri
        % 寻找四种向前路径的概率
        for j = 0 : 3
            tmp = round((j * 16) + (i - 1)/4)+1; %1 17 33 49
            x(i) = max(x(i), a(tmp));
        end
        % 取其中最大的概率
        x(i) = x(i) * 0.25 * b(i); % 转移概率矩阵A的每个值均为1/4
    end
    [~, dSRecovery(k)] = max(x);
end