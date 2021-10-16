function [v, beta]= householder(x)
n=length(x); eta=norm (x,inf); x=x/eta;
sigma=x(2:n)'*x (2:n);v=x; v(1)=1;
if sigma==0
    beta=O;
else
    alpha=sqrt(x (1)^2+sigma);
    if x(1)<=0
        v(1)=x(1)-alpha ;
    else
    v(1)=-sigma / ( x(1)+ alpha );
    end
    beta=2*v(1)^2/(sigma+v(1)^2 );
    v=v/v ( 1);
end
