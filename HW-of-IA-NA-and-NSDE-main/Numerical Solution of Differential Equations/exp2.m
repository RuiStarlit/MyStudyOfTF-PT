f = @(X,Y) -(2*pi^2*exp(pi*(X+Y)).*sin(pi*(X+Y)));
u_exact = @(X,Y)(exp(pi*(X+Y)).*sin(pi*X).*sin(pi*Y));
h = 1/64; %����
AL = diag(ones(1,62),-1);
AM = eye(63);
AR = diag(ones(1,62),1);
A = kron(AL, -eye(63))+kron(AM,-AL+4*AM-AR)+kron(AR, -eye(63));
[X, Y] = meshgrid(h:h:1-h,h:h:1-h); %ע�⵽u�ı߽�ֵΪ0
F = f(X,Y);
F = h^2*reshape(F,[],1);
% time = zeros(3,5);

% for i = 1:5
%     tic; 
    Uj = JOR(A,F,1e-5,zeros(length(F),1));
%     time(1,i)=toc;
%     tic; 
    Us = SOR(A,F,1e-5,zeros(length(F),1));
%     time(2,i)=toc;
%     tic; 
    Uc = congrad(A,F,1e-5);
%     time(3,i)=toc;
% end
Uj = reshape(Uj,63,63);
Us = reshape(Us,63,63);
Uc = reshape(Uc,63,63);
U_exact = u_exact(X,Y);
err1 = abs(Uj-U_exact);
err2 = abs(Us-U_exact);
err3 = abs(Uc-U_exact);
mesh(X,Y,err1);
title('JOR������ָ�ʽ�������ͼ');
figure;
mesh(X,Y,err2);
title('SOR������ָ�ʽ�������ͼ');
figure;
mesh(X,Y,err3);
title('�����ݶȷ�����ָ�ʽ�������ͼ');
figure;
mesh(X,Y,Uj);
title('JOR������ָ�ʽ���������ͼ');
figure;
mesh(X,Y,Us);
title('SOR������ָ�ʽ���������ͼ');
figure;
mesh(X,Y,Uc);
title('�����ݶȷ�����ָ�ʽ���������ͼ');

% U = A\F;
%U = SOR(A,F,1e-5,zeros(length(F),1));
% U = reshape(U,63,63);
% U_exact = u_exact(X,Y);
% err = abs(U-U_exact);
% mesh(X,Y,err);
% figure;
% mesh(X,Y,U);