Pi = [9/27, 16/27, 2/ 27];
Pj_i = [0  , 4/5, 1/5 ; 
       1/2, 1/2, 0   ;
       1/2, 2/5, 1/10]; % 条件分布P(j/i)
Pij = zeros(3);Pj=zeros(1,3);
for i = 1:3
    Pij(i,:) = Pj_i(i,:).*Pi(i);
end
for j =1:3
    Pj(j) = sum(Pij(j,:));
end
%信源的熵为
H=0;
for i = 1:3
    for j = 1:3
        if Pj_i(i,j) ~= 0
            H = H + -Pij(i,j)*log(Pj_i(i,j));
        end
    end
end
H = H / log(2);

HX=0;
for i = 1:3
    if Pi(i) ~= 0 
        HX= HX + -Pi(i)*log(Pi(i));
    end
end
HX = HX / log(2);

HY=0;
for j = 1:3
    if Pi(j) ~= 0 
        HY= HY + -Pi(j)*log(Pi(j));
    end
end
HY;

%当X，Y相互独立时
HXY=HX+HY;
Hmax = log(3)/ log(2);
%剩余度为
E = 1 - H/Hmax;