function [x,k]= half (a,b,tol)
%二分法
c =(a+b)/2;k=1;
m=1+round((log(b-a)-log(2*tol))/log(2));
while k<=m
    if f(c)==0
        x=c;
        return;
    elseif f(a)*f(c)<0
        b=(a+b)/2;
    else
        a=(a+b)/2;
    end
    c=(a+b)/2;x=c;k=k+1;
end
end

