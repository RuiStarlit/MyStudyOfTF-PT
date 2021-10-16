A = [0.19,0.18,0.17,0.16,0.13,0.10,0.06,0.01];
A = sort(A,'descend');
n = length(A);

%编码的元胞数组
for i=1:n
    C(i)={''};
end
Z=ones(1,n).*n;     %未分配编码的元素数量
M=1;N=n;            %指针指向当前分组首尾
%利用双重循环，分别对每个信源符号执行一次完整的Fano编码过程,
%获得当前信源符号的Fano编码
for i=1:n
    while(Z(i)>2)
        a=sum(A(1,M:N))/2;  %当前分组的总概率一半
        for K=M:N
            if sum(A(1,M:K))>=a
                if i<=K
                    char=cell2mat(C(i));
                    char=[char, '0'];
                    C(i)={char};
                    N=K;
                    Z(i)=N-M+1;
                    break;
                else
                    char=cell2mat(C(i));
                    char=[char, '1'];
                    C(i)={char};
                    M=K+1;
                    Z(i)=N-M+1;
                    break;
                end
            end
        end
    end
    
    if Z(i)==2
        if i==M
            char=cell2mat(C(i));
            char=[char '0'];
            C(i)={char};
        else
            char=cell2mat(C(i));
            char=[char '1'];
            C(i)={char};
        end
    end
    M=1;
    N=n;
end
disp('Fano编码为:')
disp(C);