function [x,k] = steffensen(x0,tol)
%Steffensen
k=0;x1=x0-(f(x0)^2)/(f(x0)-f(x0-f(x0)));
while abs(x0-x1)>tol
    x0=x1;
    x1=x0-(f(x0)^2)/(f(x0)-f(x0-f(x0)));
    k=k+1;
end
k=k+1;
x=x1;
end