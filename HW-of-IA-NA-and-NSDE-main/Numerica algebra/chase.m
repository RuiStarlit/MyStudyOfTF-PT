function x=chase (a,b,c,d)
%追赶法 (a,b,c,d) A = diag{a,b,c}, AX=d
n=length(b); h=zeros(n,1);u=h;q=h;
u(1)=c(1)/b(1); q(1)=d(1)/b(1);
x = zeros(n,1);
for i=2:n-1
    h(i)=b( i)-u(i-1)*a(i-1);
    u(i)=c(i )/h(i);
    q(i)=(d(i)-q(i-1)*a(i-1))/h(i);
end
q(n)=(d(n)-q(n-1)*a(n-1))/(b(n)-u(n-1)*a(n-1));
x(n)=q(n);
for i=n-1:-1:1
x(i)=q(i)-u(i)*x (i+1);
end