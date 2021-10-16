function I = gausscheb(f,n)              %(n+1)-pt Gauss
    x = cos(pi*(2*(0:n)'+1)/(2*n+2));    %Chebyshev points
    w = pi/(n+1) * ones(1,n+1);
    I = w*feval(f,x);                    %the integral