function I = fejer(f,n)
x = cos(pi*(2*(0:n)'+1)/(2*n+2));
y = feval(f,x)/(n+1);
g = fft(y([1:n+1 n+1:-1:1]));
hx = real(exp(2*1i*pi*(0:2*n+1)/(4*n+4)).*g');
a = hx(1:n+1); a(1) = 0.5*a(1);
w = 0*a';w(1:2:end) = 2./(1-(0:2:n).^2);
I = a*w;