m = input('码源的进制m=');
n = input('信源符号的个数n=');
l = input('各个码字的长度（请输入向量):');
sum=0;
for i = 1:n
    sum = sum + power(m,-l(i));
end
if sum<= 1
    disp('存在单义码');
else
    disp('不存在单义码');
end
