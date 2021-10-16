%chebyshev approximation
n=10;                %逼近的多项式次数
f = @(t) 1/(t-2);
syms t;
T(1:n+1)=t;         %初始化chebyshev多项式
T(1)=1;             %第一第二项chebyshev多项式
T(2)=t;
c(1:n+1)=0.0;       %初始化chebyshev系数
b(1:n+1)=0.0; 
c(1)=integral(matlabFunction(sym(f)*T(1)/sqrt(1-t^2)),-1,1)/pi;
c(2)=2*integral(matlabFunction(sym(f)*T(2)/sqrt(1-t^2)),-1,1)/pi;
for i=1:n+1
    b(i)=(-2)/(sqrt(3)*(2+sqrt(3))^(i-1));
end
C=c(1)+c(2)*t;      %初始化逼近的chebyshev多项式
for i=3:n+1
    T(i)=2*t*T(i-1)-T(i-2);     %三项递推公式
     simplify(T(i));
    T(i)=collect(T(i));
     c(i)=2*integral(matlabFunction(sym(f)*T(i)/sqrt(1-t^2)),-1,1)/pi;
    C=C+T(i)*c(i);
end
m=100;
p=linspace(-1,1,m+1);
err=zeros(m+1,1);
for i=1:m+1
    err(i)=abs(feval(matlabFunction(C),p(i))-f(p(i)));
end
plot(p,err)
ylabel('误差');
title('10次的第一类切比雪夫投影逼近误差');