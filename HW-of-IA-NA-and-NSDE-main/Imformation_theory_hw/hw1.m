img = imread('image.jpg');
img = rgb2gray(img);
[M,N] = size(img);
gray = zeros(1,256);
for i = 1:M
    for j = 1:N
        k =img(i,j)+1;
        gray(k) = gray(k)+1;
    end
end
gray = gray / (M*N);
entropy=0;
for i =1:256
    if gray(i) ~= 0
        entropy = entropy + -gray(i)*log(gray(i));
    end
end
fprintf('The entropy of this image is %f.\n',entropy)