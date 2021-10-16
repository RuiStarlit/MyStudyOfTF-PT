function dnaSeq = digit2dnaSeq(digitSignal)
% 将数字信号恢复为DNA序列
% Initialize
n = length(digitSignal);
dnaSeq = '';

for i = 1 : n
    nib2 = bitget(digitSignal(i), 6) * 2 + bitget(digitSignal(i), 5);
    
    if nib2 == 0
        dnaSeq(i) = 'A';
    elseif nib2 == 1
        dnaSeq(i) = 'C';
    elseif nib2 == 2
        dnaSeq(i) = 'G';
    else
        dnaSeq(i) = 'T';
    end
end

% Decode 2 final nibbles
nib1 = bitget(digitSignal(i), 4) * 2 + bitget(digitSignal(i), 3);
nib0 = bitget(digitSignal(i), 2) * 2 + bitget(digitSignal(i), 1);

if nib1 == 0
    dnaSeq(n+1) = 'A';
elseif nib1 == 1
    dnaSeq(n+1) = 'C';
elseif nib1 == 2
    dnaSeq(n+1) = 'G';
else
    dnaSeq(n+1) = 'T';
end

if nib0 == 0
    dnaSeq(n+2) = 'A';
elseif nib0 == 1
    dnaSeq(n+2) = 'C';
elseif nib0 == 2
    dnaSeq(n+2) = 'G';
else
    dnaSeq(n+2) = 'T';
end