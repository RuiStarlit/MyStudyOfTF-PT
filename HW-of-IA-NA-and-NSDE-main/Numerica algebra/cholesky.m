function x=cholesky(a,b)
n=length(b); v=zeros(n); x=zeros(n,1); y=zeros(n,1);
for j=1:n 
    for i=1:j-1
        v(j,i)=a(j,i)*a(i,i); 
    end
    a(j,j)=a(j,j)-a(j,1:j-1)*v(j,1:j-1)'; 
    a(j+1:n,j)=(a(j+1:n,j)-a(j+1:n,1:j-1)*v(j,1:j-1)')/a(j,j); 
end
    L=tril(a,-1)+eye(n); U=diag(diag(a))*L';
for i=1:n 
    y(i)=b(i)-L(i,1:i-1)*y(1:i-1); 
end
for j=n:-1:1
    x(j)=(y(j)-U(j,j+1:n)*x(j+1:n))/U(j,j);
end