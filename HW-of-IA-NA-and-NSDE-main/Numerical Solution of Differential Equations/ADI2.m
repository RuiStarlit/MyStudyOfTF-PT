%ADI2
f = @(X,Y) -(2*pi^2*exp(pi*(X+Y)).*sin(pi*(X+Y)));
u_exact = @(X,Y)(exp(pi*(X+Y)).*sin(pi*X).*sin(pi*Y));
h = 1/64;
[Y,X] = meshgrid(h:h:1-h, h:h:1-h);%这里y,x顺序必须要这样
F = f(X,Y);%非边界的网格点上的F函数值
time = zeros(5,1);
for k =1:5
    tic
    rho = 2^0.5-1;
    theta=h^2/(4*sin(pi*h/2)^2)*[((1-rho)/(1+rho))^1, ((1-rho)/(1+rho))^3];
    U0 = F; U1 = 0.5*F;
    U = U0; q = 0;
    while norm(U0-U1)>10^-5
        a = -theta(mod(q,2)+1)*ones(62,1)/h^2; c = a;
        b = (1+2*theta(mod(q,2)+1)/h^2)*ones(63,1);
        B1 = eye(63)-theta(mod(q,2)+1)*(-diag(ones(62,1),-1)+...
        2*diag(ones(63,1))-diag(ones(62,1),1))/h^2;
        B2 = B1;
        U0 = U1;
        U1 = U1*B2+theta(mod(q,2)+1)*F;
        for j=1:63
            U(1:63,j) = chase(a,b,c,U1(1:63,j));
        end
        U = B1*U+theta(mod(q,2)+1)*F;
        for i = 1:63
            U1(i,1:63) = chase(a,b,c,U(i,1:63))';
        end
        q = mod(q + 1,2);
    end
    time(k) = toc;
end
Uxy = U1;
UU = u_exact(X,Y);
err = abs(Uxy-UU);
subplot(2,1,1)
mesh(X,Y,Uxy)
title('ADI(双参数)法求解差分格式计算解曲面图');
subplot(2,1,2)
mesh(X,Y,err)
title('ADI(双参数)法求解差分格式计算误差图');
max(max(abs(err)))