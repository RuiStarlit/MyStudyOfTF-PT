f = @(t) exp(-t);
Iexact = exp(1)-exp(-1);
for n = 1:20
    err = abs(clenshaw_curtis(f,n) - Iexact);
    semilogy(n,err,'.r','markersize',16)
    hold on
end