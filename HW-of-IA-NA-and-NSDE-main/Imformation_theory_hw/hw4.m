%(离散)给定信源X和信源Y的联合概率分布
Pij =[       0,    0.2667,    0.0667;
        0.2963,    0.2963,         0;
        0.0370,    0.0296,    0.0074];
HXY=0;
[M,N]= size(Pij);
for i =1:M
    for j=1:N
        if Pij(i,j) ~= 0
            HXY = HXY - Pij(i,j)*log(Pij(i,j));
        end
    end
end
HXY = HXY ./ log(2);

%（连续）给定信源X和信源Y的联合概率函数f