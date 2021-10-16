%legendre approximation
n=10;                %逼近的多项式次数
f = @(t) exp(t)+sin(t);
syms t;        
P(1:n+1)=t;         %初始化Legendre多项式
P(1)=1;P(2)=t;      %第一第二项Legendre多项式
c=zeros(n+1,1);     %初始化Lengendre系数
c(1)=integral(matlabFunction(sym(f)*P(1)),-1,1)/2;
c(2)=(3/2)*integral(matlabFunction(sym(f)*P(2)),-1,1);
L=c(1)+c(2)*t;      %初始化逼近的Lengendre多项式
for i=3:n+1
    P(i)=((2*i-3)*P(i-1)*t-(i-2)*P(i-2))/(i-1);     %三项递推公式
    c(i)=((2*i-1)/2)*integral(matlabFunction(sym(f)*P(i)),-1,1);
    L=L+c(i)*P(i);
end
m=100;
p=linspace(-1,1,m+1);
err=zeros(m+1,1);
for i=1:m+1
    err(i)=abs(feval(matlabFunction(L),p(i))-f(p(i)));
end
plot(p,err)
ylabel('误差');
title('10次的勒让德投影逼近误差');