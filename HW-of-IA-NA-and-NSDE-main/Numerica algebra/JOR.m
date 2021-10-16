function x = JOR(A,b,err,x0)
%JOR迭代法 无穷范
% 
%n=length(b);x=zeros(n,1);iterations=0;
D=diag(diag(A));
L=-tril(A,-1);U=-triu(A,1);
B=D\(L+U);
r1=max(eig(B));r2=min(eig(B));
if r1<1
    w=2/(2-r1-r2);
else
    disp('需给定松弛因子');
    return;
end
x=x0-w*(D\(A*x0-b));
while norm(x-x0)>err
    %iterations=iterations+1;
    x0=x;
    x=x0-w*(D\(A*x0-b));
end