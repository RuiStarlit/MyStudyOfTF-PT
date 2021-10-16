function digitSignal = dnaSeq2Digit(dnaSeq)
%  在K-mers为3时，将DNA序列转化为数字信号

% Initialize 
n = length(dnaSeq);
digitDNASeq = zeros(1, n);

% 设A=0, C=1, G=2, T=2
% 对应的二进制为A=00; C=01; G=10; T=11
for i = 1 : n
    if dnaSeq(i) == 'A'
        digitDNASeq(i) = 0;
    elseif dnaSeq(i) == 'C'
        digitDNASeq(i) = 1;
    elseif dnaSeq(i) == 'G'
        digitDNASeq(i) = 2;
    else
        digitDNASeq(i) = 3;
    end
end

digitSignal = zeros(1, n-2);
% 将三个碱基的信息存在一个八位int整数中
% 这个数的bit的每两位代表一个碱基
% bitsll：将该数bit位向左移动n步 (Bit shift left logical)
for i = 3 : n
    digitSignal(i-2) = bitsll(digitDNASeq(i-2), 4)+...
        bitsll(digitDNASeq(i-1), 2) + digitDNASeq(i);
end