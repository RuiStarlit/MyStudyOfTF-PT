y0 = [pi; 0];
a = pi; b = 10*pi;
h = 0.01;
[t, y] = ode45(@f1,[a,b],y0);

function z = f1(t, y)
    z = zeros(2,1);
    z(1) = y(2);
    z(2) = (2/t)*y(2) - (t^-2)*y(1) + ((t^-2)-1)*sin(t) - (1+2*cos(t))/t;
end