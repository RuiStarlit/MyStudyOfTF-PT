f=@(x)1/(1+x^2)+x*0.5;
n=500;
h=2/n;
X1=zeros(n+1,1);
X2=X1;
for i=1:n+1
    X1(i)=-1+h*(i-1);           %等距节点
    X2(i)=cos((2*i-1)*pi/(2*n+2));  %第一类切比雪夫零点
end
Y1=zeros(n+1,1);
Y2=Y1;
for i=1:n+1
    Y1(i)=f(X1(i));
    Y2(i)=f(X2(i));
end
m=1000;h=2/m;
err1=zeros(m+1,1);
err2=err1;
jj=err1;
for i=1:m+1
    jj(i)=-1+h*(i-1);
end
for k=1:m+1
    L1=ones(n+1,1);
    L2=L1;
    for i=1:n+1
        for j=1:n+1
            if j~=i
           L1(i)=L1(i)*(jj(k)-X1(j))/(X1(i)-X1(j));
           L2(i)=L2(i)*(jj(k)-X2(j))/(X2(i)-X2(j));
            end
        end
    end
   err1(k)=f(jj(k))-sum(L1.*Y1);
   err2(k)=f(jj(k))-sum(L2.*Y2);
end