%Chp1 1-10
y0 = [1,exp(1)]; % 初值
tspan = [0,10];  % t范围
h = 0.01;        % 步长
time = zeros(3,1);
tic; [t1,y1] = adams(@f, @df, 0, 10, y0, h); time(1)=toc;   % 三阶Adams-Monlton
tic; [t2,y2] = RadauIA3 (@f, @df, 0, 10, y0, h);time(2)=toc;% 三阶Radau IIA
%[t3,y3] = ode45(@f,tspan,y0);
tic; [t4,y4] = pmcm (@f, 0, 10, y0, 0.001);time(3)=toc;     % 预估矫正算法
precisey = precisef(t1);                    % 精确解在每一点上的值
%z = precisef(t3);
z2 = precisef(t4);

n=length(t1);
err1=zeros(n,1);
err2=zeros(n,1);
for i = 1:n
    err1(i) = norm(precisey(i,:) - y1(i,:));
    err2(i)= norm(precisey(i,:) - y2(i,:));
end
%err3 = abs(z(:,1) - y3(:,1));
err4 = zeros(length(t4),1);
for i =1:length(t4)
    err4(i) = norm(z2(i,:) - y4(i,:));
end

subplot(2,1,1)
plot(t1,y1);
title('三阶Adams-Monlton方法计算的数值解图像');
legend('y_1','y_2');
subplot(2,1,2)
plot(t2,y2);
title('三阶Radau IIA方法计算的数值解图像');
legend('y_1','y_2');
figure;
subplot(2,1,1)
plot(t4,y4);
title('预估矫正算法方法计算的数值解图像');
legend('y_1','y_2');
subplot(2,1,2)
fplot(@(t)exp(sin(t.^2)),tspan);
hold on
fplot(@(t)exp(cos(t.^2)),tspan);
ylim([0,3]);
title('精确解图像');
legend('y_1','y_2');

figure;
subplot(2,1,1)
plot(t1,err1);
title('三阶Adams-Monlton方法计算的整体误差图');
subplot(2,1,2)
plot(t2,err2);
title('三阶Radau IIA方法计算的整体误差图');
figure;
subplot(2,1,1)
plot(t4,err4);
title('预估矫正计算的整体误差图');
function y = f(t, x)
%微分方程组函数
y = zeros(2,1);
y(1) =  2*t*x(1)*log(max(x(2),1e-3));
y(2) = -2*t*x(2)*log(max(x(1),1e-3));
end

function y = df(t,x)
%函数f对x求导的Jaccobi矩阵
y = zeros(2,2);
y(1,1) = 2*t*log(max(x(2),1e-3)); y(1,2) = 2*t*x(1)/x(2); 
y(2,1) = -2*t*x(2)/x(1); y(2,2) = -2*t*log(max(x(1),1e-3));
end

function y = precisef(t)
%精确解的函数
y = zeros(length(t), 2);
y(:,1) = exp(sin(t.^2));
y(:,2) = exp(cos(t.^2));
end