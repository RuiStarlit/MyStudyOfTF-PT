function [x,k] = secant(x0,x1,tol)
%弦截法

k=0;
while abs(x1-x0)>tol
    x2=x1-((x1-x0)/(f(x1)-f(x0))*(f(x1)));
    x0=x1;x1=x2;k=k+1;
end
x=x1;
end

