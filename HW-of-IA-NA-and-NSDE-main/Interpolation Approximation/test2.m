f = @(t) exp(t);
Iexact = 3.9774632605064226372566;
for n = 1:20
    err = abs(gausscheb(f,n) - Iexact);
    semilogy(n,err,'.r','markersize',16)
    hold on
end