function [t,y] =pmcm(f, a , b , y0 , h )
t0 = a; c1 = [0 0]'; p1=[0 0]';
A = [0   0   0;
    1/3 0   0;
    0   2/3 0];
B=[1/4 0 3/4]; c=[0 1/3 2/3]';
d = length(y0);
y0 = reshape(y0,d,1);
tc1 = t0+c(1)*h; tc2= t0+c(2)*h ; tc3=t0+c(3)*h;
Y1=y0;
Y2=y0+h*kron(A(2, 1), eye(d))*f(tc1, Y1);
Y3=y0+h*kron(A(3, 1:2), eye(d))*[f(tc1, Y1); f(tc2, Y2)];
y1=y0+h*kron(B(1:3), eye(d))*[f(tc1, Y1); f(tc2, Y2); f(tc3, Y3)];
sh = ceil((b-a)/h);
t = zeros(sh,1); y = zeros(sh,d);j =2;
t(1) = t0; y(1,:) = y0';
for n=a+2*h:h:b
    t1=t0+h; t2=t0+2*h;
    p2 = (-4)*y1+5*y0+2*h*(2*f(t1, y1)+f(t0, y0));
    m2 = p2+4*(c1-p1)/5; 
    F2 = f(t2, m2);
    c2 = y1+h*(5*F2+8*f(t1, y1)-f(t0, y0))/12;
    y2 = c2-(c2-p2)/5;
    f2 = f(t2, y2);
    t0 = t1 ;
    y0 = y1; y1 = y2;
    c1 = c2; p1 = p2 ;
    t(j) = t0; y(j,:) = y0';j = j+1;
end
