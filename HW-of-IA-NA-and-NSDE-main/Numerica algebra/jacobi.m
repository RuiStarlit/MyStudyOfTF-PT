function x = jacobi(A,b,err,x0)
% Jacobi迭代法 Jaccobi(A,b,err,x0) 无穷范
n=length(b);%x=zeros(n,1);%iterations=0;
%x0=zeros(n,1);
D=diag(diag(A));
L=-tril(A,-1);U=-triu(A,1);
B=D\(L+U);
x=B*x0+D\b;
while norm(x-x0)>err
    %iterations=iterations+1;
    x0=x;
    x= B*x0+D\b;
end