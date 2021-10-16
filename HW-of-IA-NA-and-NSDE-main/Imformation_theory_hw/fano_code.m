function C=fano_code(A)
B=sort(A,'descend');%fliplr(A);
[m,n]=size(B);
M=1;
N=n;
Z=ones(1,n).*n;

for j=1:n
    C(j)={''};
end

for i=1:n
    while(Z(i)>2)
        a=sum(B(1,M:N),2)/2;
        for K=M:N
            if sum(B(1,M:K),2)>=a
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

% celldisp(C);
end