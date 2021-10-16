%h
A = [0.19,0.18,0.17,0.16,0.13,0.10,0.06,0.01];
A = sort(A);
n = length(A);
B = A;


clc;clear;
p=[0.2 0.19 0.17 0.18 0.15 0.01 0.1];
n=length(p);
List=p;
Op_List=p;
Map=[];%Map用于进行huffman 编码,下面生成(n-1)*(n*n)的矩阵
for i=1:n-1
    Map=[Map;blanks(n)]; 
end

for i=1:n-1
    [Op_List,e]=sort(Op_List);% e 记录了原来的顺序
    
    %e(1)e(2)就是合并的两个数,小的赋1大的赋0
    Map(i,e(1))='1';
    Map(i,e(2))='0';
    %第一第二加到第二个，第一个作废
    Op_List(2)=Op_List(1)+Op_List(2);
    Op_List(1)=n;
    
    %位置还原
    Back_List=zeros(1,n);
    for j=1:n
        Back_List(e(j))=Op_List(j);
    end
    Op_List= Back_List; 
end

x=n;y=n-1;%补全Map
for i=y:-1:1
    for j=1:x
        if Map(i,j)~=' '
            for k=i-1:-1:1
                if Map(k,j)~=' '
                    for b=1:x
                        if b~=j && Map(k,b)~=' '
                            Map(k+1:y,b)=Map(k+1:y,j);
                        end
                    end
                end
            end
        end
    end
end
Map

%输出
for j=1:n
    fprintf('    概率：%.2f',p(j));
    fprintf('    哈夫曼编码：   ');
    for i=y:-1:1
       if Map(i,j)~=' '
           fprintf('%c',Map(i,j));
       end
    end
    fprintf('\n');
end

