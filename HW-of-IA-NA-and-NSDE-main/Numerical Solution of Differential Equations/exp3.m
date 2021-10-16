% exp3
tau = 0.01; h = 0.01;
r = tau/h^2;
tspan = [0,5];
n = (tspan(2)-tspan(1))/tau;
U = zeros(99,n+1);
x = h:h:1-h;
U(:,1) = sin(2*pi*x)';
a = -r*ones(98,1);c=a;b = (1+2*r)*ones(99,1);
mesh(0:tau:5,x,U);
title('解曲面图');
%% Matlab内置求解一维抛物型和椭圆型 PDE函数
m = 0;
xmesh = 0:h:1;
ttspan = 0:tau:5;
sol = pdepe(m,@pdefun,@icfun,@bcfun,xmesh,ttspan);
u = sol(:,:,1)';
figure;
mesh(ttspan,xmesh,u)
title('pdepe的解曲面图');
xlabel('x')
ylabel('t')
zlabel('u(x,t)')
max(max(abs(U-u(2:100,:))));
function [c,f,s] = pdefun(x,t,u,dudx)
    c = 1;
    f = dudx - 0.5*u.^2;
    s = 0;
end

function u0 = icfun(x)
    u0 = sin(2*pi*x);
end

function [pl,ql,pr,qr] = bcfun(xl,ul,xr,ur,t)
  pl = ul;
  ql = 0;
  pr = ur;
  qr = 0;
end