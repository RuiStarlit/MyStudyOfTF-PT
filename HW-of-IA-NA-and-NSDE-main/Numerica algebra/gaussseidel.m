function [x,iterations] = gaussseidel(A,b,err,x0)
%Gauss-Seidel迭代法 gaussseidel(A,b,err,x0) 无穷范
n=length(b);x=zeros(n,1);iterations=0;
if nargin==3
  x0=zeros(n,1);
  
D=diag(diag(A));
L=tril(A,-1);U=triu(A,1);
B=-inv(D+L)*U;
x=B*x0+(D+L)\b;
while norm(x-x0,inf)>err
    iterations=iterations+1;
    x0=x;
    x=B*x0+(D+L)\b;
end
end

