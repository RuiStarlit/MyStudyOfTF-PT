function z = Gauss ( a, b, y0, d, h)
% 二级Gauss方法
A = [1/4 1/4-sqrt(3)/6;
    1/4+sqrt(3)/6 1/4];
B = [1/2 1/2] ;
c = [1/2-sqrt(3)/6; 1/2+sqrt(3)/6 ];
A1 = [0   0   0;
    1/3 0   0;
    0   2/3 0];
B1 = [1/4 0 3/4]; t0=a; s=2;
for n = a+h:h:b
    err1 = 1; err2 = 1; t=kron(ones(s,1), t0)+h*c ;
    tc11 = t0; tc12 =t0+c(1)*h/3; tc13 = t0+2*c(1)*h/3;
    Y11=y0 ; Y12 = y0+h*c(1)*kron(A1(2, 1), eye(d))*f(tc11, Y11);
    Y13=y0+h*c(1)*kron(A1(3, 1:2), eye(d))*[f(tc11, Y11); f(tc12, Y12)] ;
    Y10=y0+h*c(1)*kron(B1(1:3), eye(d))*[f(tc11, Y11); f(tc12, Y12); f(tc13, Y13)];
    
    tc21=t0; tc22=t0+c(2)*h/3; tc23=t0+2*c(2)*h/3;
    
    Y21=y0; Y22=y0+h*c(2)*kron(A1(2,1), eye(d))*f(tc21, Y21);
    Y23=y0+h*c(2)*kron(A1(3, 1:2) , eye(d))*[f(tc21, Y21); f(tc22, Y22)];
    Y20=y0+h*c(2)*kron(B1(1:3), eye(d))*[f(tc21, Y21); f(tc22, Y22); f(tc23, Y23)];
    Y0=[Y10; Y20];
    while err1 >=1e-12 && err2 >=1e-12
        F=[f(t(1), [Y0(1); Y0(2)]); f(t(2), [Y0(3); Y0(4)])];
        r=Y0-kron(ones(s, 1), y0)-h*kron(A, eye(d))*F;
        dF=[df( t(1), [Y0(1); Y0(2)])   zeros(2,2);
            zeros(2,2)                  df( t(2), [Y0(3);Y0(4)])];
        Y1=Y0-(eye(s*d)-h*kron(A, eye(d))*dF) \ r;
        err1 = norm(Y1-Y0); err2 = norm(r); Y0 = Y1;
    end
    y1=y0+h*kron(B, eye(d))*[f(t(1), [Y0(1); Y0(2)]) ; f(t(2), [Y0(3); Y0(4)])];
    t0 = t0+h; y0=y1;
end
z = y1;