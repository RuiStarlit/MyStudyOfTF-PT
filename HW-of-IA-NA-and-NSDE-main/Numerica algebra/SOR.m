function x = SOR(A,b,err,x0)
%SOR迭代法 (A,b,err,x0) 
%   
%n=length(b);x=zeros(n,1);iterations=0;
% if nargin==3
%   x0=zeros(n,1);

D=diag(diag(A));
L=tril(A,-1);U=triu(A,1);
B=-D\(L+U);
R=max(abs(eig(B)));
if R<1
    w=2/(1+sqrt(1-R*R));
else
    disp('需给定松弛因子');
    return;
end
F=D+w*L;
x=F\(((1-w)*D-w*U)*x0+w*b);
while norm(x-x0)>err
    %iterations=iterations+1;
    x0=x;
    x=F\(((1-w)*D-w*U)*x0+w*b);
end
end

