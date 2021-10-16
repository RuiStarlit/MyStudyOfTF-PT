% simulate fungi competetion model with cellular automata
%e1 = input('Please input fungi A"s Extension rate:');
%e2 = input('Please input fungi B"s Extension rate: ');
%d1 = input('Please input fungi A"s Decomposition rate:');
%d2 = input('Please input fungi B"s Decomposition rate: ');
load('fungi.mat');
n=1000;                                              %size of wood is n*n;
f1 = 7; f2 = 8;   %Choose the number of the two fungi
UL = [n 1:n-1]; DR = [2:n 1];                        % Up and Left neighbor; Down and Right neighbor.
veg=zeros(n,n); E=0;A=1;B=2;                         % veg: empty=0 fungi A=1 fungi B=2
e1 = Extension(1,f1);e2 = Extension(1,f2) ;              % Extension rate 
t1= e1*100; t2=e2*100; t1=int32(t1); t2=int32(t2);
a = gcd( t1 , t2); b=t1/a;c=t2/a;                    % Discretization time
d1 = Decompostion(1,f1); d2 = Decompostion(1,f2);          %Decompostion rate
s1 = d2/d1;  s2 = d1/d2;                             % The assumed competition coefficient
D = ones(n);
imh = image(cat(3,D,D,D));                           % color of empty is white
veg(250,500)=1; veg(750,500)=2;                      % Fungal colonization.
j=1;d=max(b,c); z=0;Z=0;                             % some  state variable.
if b==d
    z=1;
end
day=0; k=0;f=k; area_a=zeros(122,1);area_b=zeros(122,1);
while(day<122)
   j=j+1;   
    isE = (veg==E); isA = (veg==A); isB = (veg==B);
    if j==b
        Z=1;
        if z
            j=1;k=k+1;
        end
    end
    if j ==c
        Z=2;
        if ~z
            j=1;k=k+1;
        end
    end
    switch Z
        case 1
            sums =            (veg(UL,:)==A) + ...
                (veg(:,UL)==A)  +   (veg(:,DR)==A) + ...  %Find out if there's any A fungus in neighborhood.
                                   (veg(DR,:)==A);
            veg = veg + ((isE) & sums>0)-( (isB) & rand(n,n) > s1/(s1+s2)*ones(n) & sums>0 );
                                                                            %update veg matrix
        case 2
            sums =            (veg(UL,:)==B) + ...
                (veg(:,UL)==B)  +   (veg(:,DR)==B) + ...  %Find out if there's any B fungus in neighborhood.
                                   (veg(DR,:)==B);                  %update veg matrix
            veg = veg + 2*((isE) & sums>0) + ( (isA) & rand(n,n) > s2/(s1+s2)*ones(n) & sums>0 );
        otherwise
    end
    Z=0; 
    set(imh, 'cdata', cat(3,isE+isA,isE+isB,isE ))
    drawnow
    if k == d*a % a new day
        day=day+1;k=0;
        disp(['day:',num2str(day)]);
        %caculate area proportion (%)
        aa = sum(isA,'all'); ab = sum(isB,'all');
        area_a(day)=aa/10e3;area_b(day)=ab/10e3;
    end
end
figure(2);
plot(1:day,area_a,1:day,area_b);
title('The area proportion of Fungi');
legend(cell2mat(name(f1)),cell2mat(name(f2)));