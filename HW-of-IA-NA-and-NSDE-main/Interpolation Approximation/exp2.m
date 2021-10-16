%Exp 2
f=@(x) 1/(1+x^2)+x*0.5;             % Approximated function
n=100;h=2/n;                        %插值的节点数量
X1=zeros(n+1,1);
X2=X1;Y1=X1;Y2=X2;err1=X1;err2=X1;
for i=1:n+1
    X1(i)=-1+h*(i-1);               %等距节点
    X2(i)=cos((2*i-1)*pi/(2*n+2));  %第一类切比雪夫零点
end
for i=1:n+1                         %节点对应函数值
    Y1(i)=f(X1(i));
    Y2(i)=f(X2(i));
end
%进行拉格朗日形式插值 由于进行符号函数运算会浪费很多运算和时间，于是直接进行节点的数值计算。
m=500;                             %计算误差的节点数量
t=zeros(m+1,1);h2=2/m;
for i=1:m+1
    t(i)=-1+h2*(i-1);               %计算误差的节点
end
for k=1:m+1
    Lag1=ones(n+1,1);Lag2=Lag1;
    for i=1:n+1
        for j=1:n+1                 %该节点处的拉格朗日基函数
            if j~=i
           Lag1(i)=Lag1(i)*(t(k)-X1(j))/(X1(i)-X1(j));
           Lag2(i)=Lag2(i)*(t(k)-X2(j))/(X2(i)-X2(j));
            end
        end
    end
   err1(k)=f(t(k))-sum(Lag1.*Y1);   %sum(Lag1.*Y1)即拉格朗日形式插值多项式在该点的值
   err2(k)=f(t(k))-sum(Lag2.*Y2);
end
plot(t,err1)
ylabel('误差');
title('基于等距节点的于拉格朗日形式插值多项式逐点误差图');
grid on
