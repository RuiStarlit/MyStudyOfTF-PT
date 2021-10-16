z=adams(a ,b ,y0 ,d ,h) 
t0=a; alpha=[0 -1 1]'; beta=[-1 8 5]'/12;
A=[0 0 0;1/3 0 0;0 2/3 0]; B=[1/4 0 3/4]; c=[0 1/3 2/3]'; 
tc1=t0+c(1)*h; tc2=t0+c(2)*h; tc3=t0+c(3)*h; 
Y1=y0; Y2=y0+h*kron(A(2 ,1) , eye(d))*f ( tc1 ,Y1); 
Y3=y0+h*kron(A(3 ,1:2) , eye(d))*[ f ( tc1 ,Y1); f ( tc2 ,Y2)]; 
y1=y0+h*kron(B(1:3) , eye(d))*[ f ( tc1 ,Y1); f ( tc2 ,Y2);f ( tc3 ,Y3)]; 
for n=a+2*h:h:b 
t1=t0+h; t2=t0+2*h; tc1=t0 +(c(1)+1)*h; 
tc2=t0 +(c(2)+1)*h; tc3=t0 +(c(3)+1)*h; 
Y1=y1; Y2=y1+h*kron(A(2 ,1) , eye(d))*f ( tc1 ,Y1); 
Y3=y1+h*kron(A(3 ,1:2) , eye(d))*[ f ( tc1 ,Y1); f ( tc2 ,Y2)];
y20=y1+h*kron(B(1:3) , eye(d))*[ f ( tc1 ,Y1); f ( tc2 ,Y2); f ( tc3 ,Y3)]; 
w=h*(beta (1)*f (t0 , y0)+beta (2)*f (t1 ,y1))-( alpha (1)*y0+alpha (2)*y1 ); 
err1 =1; err2 =1; 
while err1 >=10?(-12) & err2 >=10^(-12) 
r=y20-h*beta (3)*f (t2 , y20)-w; 
y21=y20-inv(eye(d)-h*beta (3)*df (t2 , y20))*r ; 
err1=norm(y21-y20 ); err2=norm( r ); 
y20=y21; 
end 
y2=y20; t0=t0+h; y0=y1; y1=y2; 
end
