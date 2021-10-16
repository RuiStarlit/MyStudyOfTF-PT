function x = qrfact (A,b)
n=length(A);Q=eye(n); R=zeros(n);
for j=1:n
    if j<n
        [v,beta]= householder(A(j:n,j));
    else
        v=1; beta=2-2*mod(n,2);
    end
    A(j:n,j :n)=(eye(n-j+1)- beta*v*v')*A(j:n,j :n);
    d(j)= beta ;
    if  j<n
    A(j+1:n,j)=v(2:n-j+1);
    end
end
R=triu(A);
for k=1:n
    H=eye(n);
    H1=eye(n-k+1)-d(k)*[1,A(k+1:n,k)']'*[1,A(k+1:n,k)'];
    H(k :n,k:n)=H1;Q=Q*H;
end
R,Q
x=zeros (n ,1);b=Q'*b ;
for j=n:-1:1
x(j)=(b(j)-sum(R(j,j+1:n).*x(j+1:n)'))/R(j ,j );
end


