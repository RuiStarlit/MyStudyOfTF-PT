function [ X1 ] = ex6f( X )
% µÚÁùÕÂPicardµü´úº¯Êý
%   
 x0=X(1);y0=X(2);z0=X(3);
 x1 = (2*cos(y0*z0)+1)/6;
 y1 = (sqrt(x0^(2)+sin(z0)+1.06))/9-0.1;
 z1 = -(3*exp(-x0*y0)+10*pi-3)/60;
 X1(1,1)=x1;X1(2,1)=y1;X1(3,1)=z1;
end

