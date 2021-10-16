function [X,k] = sdescent(A,b,err)
%最速下降法 inf
%   
n = length(b);X0=zeros(n,1);r0=b-A*X0;k=0;
while norm(r0,inf)>=err
    k=k+1;alpha0=r0'*r0/(r0'*A*r0);
    X1=X0+alpha0*r0;r0=b-A*X1;X0=X1;
end
X=X0;
end

