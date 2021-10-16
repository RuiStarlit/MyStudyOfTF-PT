function z=gill (a, b, y0,d ,h) 
% 四级gill方法
A = [0              0              0             0; 
     1/2            0              0             0; 
     (sqrt(2)-1)/2  (2-sqrt(2))/2  0             0;
     0              -sqrt(2)/2     (2+sqrt(2))/2 0]; 
c = [0              1/2            1/2           1]';
B = [1/6            (2-sqrt(2))/6  (2+sqrt(2))/6 1/6]; t0=a;
for n=a+h:h:b
    t1=t0+c(1)*h; t2=t0+c(2)*h; t3=t0+c(3)*h; t4=t0+c(4)*h;
    Y1=y0; Y2=y0 + h*kron(A(2, 1), eye(d))* f(t1, Y1);
    Y3=y0 + h*kron(A(3, 1:2), eye(d ))*[f(t1, Y1); f(t2, Y2)];
    Y4=y0 + h*kron(A(4, 1:3), eye(d))*[f(t1, Y1); f(t2, Y2); f(t3, Y3)];
    y1=y0 + h*kron(B(1:4), eye(d))*[f(t1, Y1);f(t2, Y2); f(t3, Y3); f(t4,Y4)];
    t0=t0+h; y0=y1 ;
end
z = y1