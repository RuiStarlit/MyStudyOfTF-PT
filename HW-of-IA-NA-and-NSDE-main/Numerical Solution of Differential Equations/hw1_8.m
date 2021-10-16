% 取步长h=0.01 计算二阶初值问题
y0 = [pi; 0];
a = pi; b = 10*pi;
h = 0.01;
[t,y1] = kutta (@f, a, b, y0, h);
y2 = precisef(t);
subplot(2,1,1);
plot(t,y1(:,1));
title('数值解 x_n 的曲线图')
xlabel('t');

subplot(2,1,2);
err = abs(y1(:,1)-y2(:,1));
plot(t, err);
title('整体误差图 |x(t_n)-x_n| ')
xlabel('t');

%微分方程组函数
function z = f(t, y)
    z = zeros(2,1);
    z(1) = y(2);
    z(2) = (2/t)*y(2) - (t^-2)*y(1) + ((t^-2)-1)*sin(t) - (1+2*cos(t))/t;
end

%精确解的函数
function y = precisef(t)
y = zeros(length(t), 2);
y(:,1) = t + sin(t);
y(:,2) = 1 + cos(t);
end