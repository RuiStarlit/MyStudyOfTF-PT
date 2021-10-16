function x=cholesky1 (A,b)
n=length(b); L=zeros(n);x =zeros(n,1); y =zeros(n,1);
L(1,1)=sqrt(A(1,1));
for i=2:n
    L(i,1)=A(i,1)/L(1,1);
end
for k=2:n
    L(k,k)=sqrt(A(k,k)-sum(L(k,1:k-1).*L(k,1:k-1)));
    for i=k+1:n
         L(i,k)=(A(i,k)-sum(L(i,1:k-1).*L(k,1:k-1)))/L(k,k);
    end
end
L=tril(L);
for i=1:n
    y(i)=(b(i)-sum(L(i,1:i-1).*y(1:i-1)'))/L(i,i);
end
for j=n:-1:1
    x(j)=(y(j)-sum(L(j+1:n,j)'.*x(j+1:n)'))/L(j,j);
end