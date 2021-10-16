function [X,k] = precongrad(A,b,M,err)
% 预优共轭梯度法 inf
n=length(b); X0=zeros (n ,1);r0=b-A*X0;k=0;
zeta0=M\r0;rho0=r0'*zeta0;P0=zeta0 ;
while norm(r0,inf)>=err
    k=k+1;
    omega=A*P0; alpha=rho0/(P0'* omega); X1=X0+alpha *P0;
    r1=r0-alpha *omega; zeta1 =M\r1 ; rho1=r1'*zeta1 ;
    beta=rho1 /rho0; P1=zeta1+beta*P0;
    r0=r1;X0=X1;P0=P1; rho0=rho1 ;
end
X=X0;
end
