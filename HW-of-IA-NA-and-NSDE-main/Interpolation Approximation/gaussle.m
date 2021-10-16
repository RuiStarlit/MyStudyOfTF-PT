function I = gaussle(f,n)                  
    %(n+1)-pt Gauss -legendre
    beta = 0.5./sqrt(1-(2*(1:n)).^(-2)); %3-term recurrence
    T = diag(beta,1) + diag(beta,-1);    %Jacobi matrix PS：alpha of legendre is zero
    [V,D] = eig(T);                      %eigenvalue decomposition
    x = diag(D); [x,i] = sort(x);        %nodes (= Legendre zero point)
    w = 2*V(1,i).^2;                     %weights
    y=zeros(n+1,1);                      %don't konw how to use feval function
    for j=1:n+1
        y(j)=f(x(j));
    end
    I = w*y;                             %the integral