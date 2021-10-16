err=10^(-8);
%Picard start
syms x
syms y
syms z
F=[ (2*cos(y*z)+1)/6;
    (sqrt(x^(2)+sin(z)+1.06))/9-0.1;
    -(3*exp(-x*y)+10*pi-3)/60; ];
X0=[0;0;0];x=X0(1);y=X0(2);z=X0(3);
X1=eval(F);k=1;
while norm(X1-X0,2)>err
   X0=X1;x=X0(1);y=X0(2);z=X0(3);
   X1=eval(F);k=k+1;
end
%Picrad end

%计算Picrad加速矩阵 start
syms x
syms y
syms z
F=[ (2*cos(y*z)+1)/6;
    (sqrt(x^(2)+sin(z)+1.06))/9-0.1;
    -(3*exp(-x*y)+10*pi-3)/60; ];
G=jacobian(F,[x;y;z]);x=0;y=0;z=0;
P=eval(G);
B=(eye(3)-P)^-1;
% end
%Picard加速迭代法
X0=[0;0;0];x=X0(1);y=X0(2);z=X0(3);
X1=B*(eval(F)-P*X0);k=1;
while norm(X0-X1,2)>err
   X0=X1;x=X0(1);y=X0(2);z=X0(3);
   X1=B*(eval(F)-P*X0);k=k+1;
end
X1
%end

%Newton迭代法
syms x
syms y
syms z
F=[ 6*x-2*cos(y*z)-1;
    sqrt(x^(2)+sin(z)+1.06)-9*(y+0.1);
    3*exp(-x*y)+60*z+10*pi-3; ];
G=jacobian(F,[x;y;z]);
X0=[0;0;0];x=X0(1);y=X0(2);z=X0(3);
X1=X0-(eval(G))\(eval(F));k=1;
while norm(X0-X1,2)>err
   X0=X1;x=X0(1);y=X0(2);z=X0(3);
   X1=X0-(eval(G))\(eval(F));k=k+1;
end
X1
%end

