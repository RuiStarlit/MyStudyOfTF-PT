A=[ 16 2 3 1;
    1 10 4 3;
    3 1 15 2;
    1 2 4 18;];
b=[15 1 -40 61]';err=10^(-5);n=length(b);
b=A'*b;
A=A'*A;
time=zeros(4,1);anwser=cell(4,1);iterations=zeros(4,1);
tic
[anwser{1,1},iterations(1)]=jacobi(A,b,err);
toc
time(1)=toc;
tic
[anwser{2,1},iterations(2)]=gaussseidel(A,b,err);
toc
time(2)=toc;
tic
[anwser{3,1},iterations(3)]=JOR(A,b,err);
toc
time(3)=toc;
tic
[anwser{4,1},iterations(4)]=SOR(A,b,err);
toc
time(4)=toc;