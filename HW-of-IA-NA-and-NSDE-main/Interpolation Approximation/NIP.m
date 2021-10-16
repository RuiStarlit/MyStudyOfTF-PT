function [f] = NIP (x,y)        
%NIP Newton interpolation
 n=length(x);
 syms p;
N=zeros(n);                     %差商表 下三角
N(1,1)=y(1);
for i=2:n
    N(i,1) = y(i);       
    for j = 2:i           
        N(i,j) = (N(i,j-1)-N(i-1,j-1))/(x(i)-x(i-j+1));    
    end
end
f=y(1);
for i=1:n-1                     %构造插值函数
    t=N(i+1,i+1);
    for j =1:i
        t=t*(p-x(j));
    end
    f=f+t;
   % simplify(f);
end
f=subs(f,'p','x');
f=collect(f);