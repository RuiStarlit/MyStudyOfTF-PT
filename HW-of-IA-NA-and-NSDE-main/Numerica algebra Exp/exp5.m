A=zeros(50,50);b=zeros(50,1);
for i=1:50
    for j=1:50
        if i == j
            A(i,j)=50*i;
        else
            A(i,j)=max(i,j);
        end
    end
    for j=1:50
        b(i)=b(i)+A(i,j)*(51-j);
    end
end
time=zeros(3,1);anwser=cell(3,1);iterations=zeros(3,1);
M=diag(diag(A));err=10^(-8);
tic
[anwser{1,1},iterations(1)]=sdescent(A,b,err);
toc
time(1)=toc;
tic
[anwser{2,1},iterations(2)]=congrad(A,b,err);
toc
time(2)=toc;
tic
[anwser{3,1},iterations(3)]=precongrad(A,b,M,err);
toc
time(3)=toc;