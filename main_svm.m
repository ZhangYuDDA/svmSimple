% 本程序有三个变量：
% 1. 核参数类型 kerneltype，可选值为‘liear’或者‘rbf’，linear表示使用线性核，rbf表示使用高斯核
% 2. C值
% 3. 核参数 delta，如果kerneltype为rbf，则需要设置delta值；如果kerneltype为linear，则不需要设置delta值

clear all;
close all;
C = 10;  % 这里设置C值
%kertype = 'linear';  % 这里使用linear kernel
kertype = 'rbf';


%产生随机数据
%n = 50;
%randn('state',6);
%rng('default');
%rng(2);
%x1 = randn(2,n);    %2行N列矩阵
%x1 = [x1(1,:)+4;x1(2,:)];
%y1 = ones(1,n);       %1*N个1
%x2 = 3+randn(2,n);   %2*N矩阵
%x2 = [x2(1,:);x2(2,:)+6];
%y2 = -ones(1,n);      %1*N个-1

%load('data1.mat');
load('data2.mat');
X = X';
y = y';
n = length(y);
randSeq = randperm(n);

%生成训练数据和测试数据
xTest = X(:,randSeq(1:floor(n*0.2)));
yTest = y(randSeq(1:floor(n*0.2)));
X = X(:,randSeq((floor(n*0.2)+1):n));
y = y(randSeq((floor(n*0.2)+1):n));

%把随机产生的数据画出来
figure;
%plot(x1(1,:),x1(2,:),'bx',x2(1,:),x2(2,:),'k.'); 
%axis([-3 8 -3 9]);

x1 = X(:,find(y == 1));
x2 = X(:,find(y == -1));
y1 = y(find(y == 1));
y2 = y(find(y == -1));

plot(x1(1,:),x1(2,:),'bx',x2(1,:),x2(2,:),'k.'); 
hold on;
pause;

kertype = 'rbf';

% 训练svm分类器
X = [x1,x2];        %训练样本d*n矩阵，n为样本个数，d为特征向量个数。 X是训练样本
Y = [y1,y2];        %训练目标1*n矩阵，n为样本个数，值为+1或-1。Y是训练样本的标签，+1代表该样本属于正类，-1代表该样本属于负类
svm = svmTrain(X,Y,kertype,C);
plot(svm.Xsv(1,:),svm.Xsv(2,:),'ro'); % 画出分类边界
pause;

% 测试svm分类器
%[x1,x2] = meshgrid(-2:0.05:7,-2:0.05:7);  %x1和x2都是181*181的矩阵
%[x1,x2] = meshgrid(0:0.05:4.5,1:0.05:5);
[x1,x2] = meshgrid(0:0.02:1,0.4:0.02:1);
[rows,cols] = size(x1);  
nt = rows*cols;                  
Xt = [reshape(x1,1,nt);reshape(x2,1,nt)];
Yt = ones(1,nt);
result = svmTest(svm, Xt, Yt, kertype);
Yd = reshape(result.Y,rows,cols);
contour(x1,x2,Yd,'y');


%训练精度
trainResult = svmTest(svm,X,Y,kertype);
fprintf('训练精度为%f\n',trainResult.accuracy);

%测试精度
testResult = svmTest(svm,xTest,yTest,kertype);
fprintf('测试精度为%f\n',testResult.accuracy);



