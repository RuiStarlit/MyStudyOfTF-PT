%给定两个分布列，P(i)代表X发送第i的消息的先验概率
Pi = [9/27, 16/27, 2/ 27];
%Pi_j 代表P(i/j)，当第j个消息被接受时，发送的是第i个消息的后验概率
Pi_j=[
    0,         0.4500,    0.9000;
    0.8889,    0.5000,         0;
    0.1111,    0.0500,    0.1000];
Pij =[
         0    0.2667    0.0667
    0.2963    0.2963         0
    0.0370    0.0296    0.0074];
[M,N]= size(Pi_j);
%互信息量Icp(i,j)
Icp = zeros(3);
for i =1:M
    for j=1:N
        if (Pi_j(i,j) ~= 0) && (Pi(i) ~=0 )
            Icp(i,j) = log(Pi_j(i,j)/Pi(i));
        else
            Icp(i,j)=0;
        end
    end
end
Icp = Icp ./ log(2);
%平均互信息量
IcpXY = 0;
for i =1:M
    for j=1:N
        if Pij(i,j) ~= 0
            IcpXY = IcpXY + Pij(i,j)*Icp(i,j);
        end
    end
end
HX=0;HX_Y=0;
for i = 1:length(Pi)
    if Pi(i) ~= 0 
        HX= HX + -Pi(i)*log(Pi(i));
    end
end
for i =1:M
    for j=1:N
        if Pi_j(i,j) ~= 0 
        HX_Y=HX_Y - Pij(i,j)*log(Pi_j(i,j));
        end
    end
end
IcpXY_=(HX-HX_Y) ./ log(2);