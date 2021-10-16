function I = clenshaw_curtis(f,n)
%Clenshaw_curtis 积分公式
%   
x = cos(pi*(0:n)'/n);
fx = feval(f,x)/(2*n);
g = real(fft(fx([1:n+1 n:-1:2])));
a = [g(1); g(2:n)+g(2*n:-1:n+2); g(n+1)];
w = 0*a'; w(1:2:end) = 2./(1-(0:2:n).^2);
I = w*a;
end

