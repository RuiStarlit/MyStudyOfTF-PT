function I = gaussla(f,n)                  
    % Gauss -Laguerre
    alpha=2*(1:n+1)-1;
    beta = 1:n;                                      %3-term recurrence
    T = diag(beta,1) + diag(beta,-1)+diag(alpha);    %Jacobi matrix 
    [V,D] = eig(T);                                  %eigenvalue decomposition
    x = diag(D); [x,i] = sort(x);                    %nodes (= Legendre zero point)
    w = V(1,i).^2;                                   %weights
    y=zeros(n+1,1);                                  %don't konw how to use feval function
    for j=1:n+1
        y(j)=f(x(j));
    end
    I = w*y;                                         %the integral