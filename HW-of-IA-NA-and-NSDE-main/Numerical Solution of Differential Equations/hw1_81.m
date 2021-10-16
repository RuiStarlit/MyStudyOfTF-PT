A = [ 0  0 0;
     1/2 0 0;
     -1  2 0];
B = [1/6 2/3 1/6];
C = [0 1/2 1];

l =2;
sum = 0;
%for i =1:3
  %  sum = sum + B(i)*power(C(i),l-1);
%end
j=1;
for i = 1:3
    sum=sum+A(i,j)*power(C(i),l-1)*B(i);
end
disp(B(j)*(1-(power(C(j),l)))/l);
disp(sum);