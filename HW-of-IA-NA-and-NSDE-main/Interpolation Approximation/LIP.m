function [f] = LIP( x,y )   
%lagrange interpolation
 n=length(x);
 syms p;
 f=0;
 for i=1:n
    lag=y(i);
    for j=1:n               %primary function
        if j~=i
            lag=lag*(p-x(j))/(x(i)-x(j));  
        end
    end
    f=f+lag;
    simplify(f);
 end
  f=subs(f,'p','x');
  f=collect(f);
  f=vpa(f,6);               %系数的精度
end

