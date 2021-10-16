function [t,y] = kutta (f,a, b, y0, h) % [a,b] y0 h
% 三级Kutta方法
d = length(y0);
y0 = reshape(y0,d,1);
A = [0    0    0; 
     1/2  0    0; 
     -1   2    0]; 
c = [0    1/2  1]';
B = [1/6  2/3  1/6]; t0=a;
sh = ceil((b-a)/h);
t = zeros(sh,1); y = zeros(sh,d);j =1;
for n=a+h:h:b
    t(j) = t0; y(j,:) = y0';
    t1=t0+c(1)*h; t2=t0+c(2)*h; t3=t0+c(3)*h;
    Y1=y0;
    Y2=y0 + h*kron(A(2, 1), eye(d))* f(t1, Y1);
    Y3=y0 + h*kron(A(3, 1:2), eye(d ))*[f(t1, Y1); f(t2, Y2)];
    y1=y0 + h*kron(B(1:3), eye(d))*[f(t1, Y1); f(t2, Y2); f(t3, Y3)];
    t0=t0+h; y0=y1; j = j+1;
end
t(j)=t0; y(j,:)=y0;
end

