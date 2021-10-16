n=5;
f = @(t) 1/(t-2);
syms t;
T(1:n+1)=t;
T(1)=1;
T(2)=t;
c(1:n+1)=0.0;
b(1:n+1)=0.0;
b(1)=integral(matlabFunction(sym(f)*T(1)/sqrt(1-t^2)),-1,1)/pi;
b(2)=2*integral(matlabFunction(sym(f)*T(2)/sqrt(1-t^2)),-1,1)/pi;
C=c(1)+c(2)*t;
for i=1:n+1
    c(i)=-2/(sqrt(3)*(2+sqrt(3))^(i-1));
end
for i=3:n+1
    T(i)=2*t*T(i-1)-T(i-2);
     b(i)=2*integral(matlabFunction(sym(f)*T(i)/sqrt(1-t^2)),-1,1)/pi;
    C=C+T(i)*c(i);
end
% err=matlabFunction(sym(f)*C);
% fplot(err,[-1,1])