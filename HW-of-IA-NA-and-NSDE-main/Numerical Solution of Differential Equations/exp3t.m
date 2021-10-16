h = 0.01; tau = 0.01;
m = 0;
xmesh = 0:h:1;
ttspan = 0:tau:5;
sol = pdepe(m,@pdefun,@icfun,@bcfun,xmesh,ttspan);
u = sol(:,:,1)';
mesh(ttspan,xmesh,u)
xlabel('x')
ylabel('t')
zlabel('u(x,t)')
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
