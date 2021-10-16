h=0.1;q=-4;r=zeros(10,1);
b=zeros(9,1);a=zeros(8,1);c=zeros(8,1);d=zeros(9,1);
e=zeros(9);
for i=1:9
    b(i)=2+h*h*q;
end
for i=1:8
    a(i)=-1;c(i)=-1;
    r(i)=(4-pi*pi)*(2*cos(pi*h*i)+3*sin(pi*h*i));
end
r(9)=(4-pi*pi)*(2*cos(pi*h*9)+3*sin(pi*h*9));
d(1)=(-1)*h*h*r(1)+2;
d(9)=(-1)*h*h*r(9)-2;
for i=2:8
    d(i)=(-1)*h*h*r(i);
end
for i=1:8
    e(i,i)=b(i);
    e(i+1,i)=a(i);
    e(i,i+1)=c(i);
end
e(9,9)=b(9);
y=zeros(9,1);
for i=1:9
    y(i)=2*cos(pi*h*i)+3*sin(pi*h*i);
end
time=zeros(4,10);
tic
A=lux(e,d);
toc
t(1)=toc;
tic
B=cholesky(e,d);
toc
t(2)=toc;
tic
C=qrfact(e,d);
toc
t(3)=toc;
tic
D=chase(a,b,c,d);
toc
t(4)=toc;
err=[abs(y(1)-A(1)),abs(y(1)-B(1)),abs(y(1)-C(1)),abs(y(1)-D(1))];
for i=2:9
   if abs(y(i)-A(i)) > err(1)
       err(1)=abs(y(i)-A(i));
   end
   if abs(y(i)-B(i)) > err(2)
       err(2)=abs(y(i)-B(i));
   end
   if abs(y(i)-C(i)) > err(3)
       err(3)=abs(y(i)-C(i));
   end
   if abs(y(i)-D(i)) > err(4)
       err(4)=abs(y(i)-D(i));
   end
end
