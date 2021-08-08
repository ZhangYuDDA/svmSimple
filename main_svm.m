% ������������������
% 1. �˲������� kerneltype����ѡֵΪ��liear�����ߡ�rbf����linear��ʾʹ�����Ժˣ�rbf��ʾʹ�ø�˹��
% 2. Cֵ
% 3. �˲��� delta�����kerneltypeΪrbf������Ҫ����deltaֵ�����kerneltypeΪlinear������Ҫ����deltaֵ

clear all;
close all;
C = 10;  % ��������Cֵ
%kertype = 'linear';  % ����ʹ��linear kernel
kertype = 'rbf';


%�����������
%n = 50;
%randn('state',6);
%rng('default');
%rng(2);
%x1 = randn(2,n);    %2��N�о���
%x1 = [x1(1,:)+4;x1(2,:)];
%y1 = ones(1,n);       %1*N��1
%x2 = 3+randn(2,n);   %2*N����
%x2 = [x2(1,:);x2(2,:)+6];
%y2 = -ones(1,n);      %1*N��-1

%load('data1.mat');
load('data2.mat');
X = X';
y = y';
n = length(y);
randSeq = randperm(n);

%����ѵ�����ݺͲ�������
xTest = X(:,randSeq(1:floor(n*0.2)));
yTest = y(randSeq(1:floor(n*0.2)));
X = X(:,randSeq((floor(n*0.2)+1):n));
y = y(randSeq((floor(n*0.2)+1):n));

%��������������ݻ�����
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

% ѵ��svm������
X = [x1,x2];        %ѵ������d*n����nΪ����������dΪ�������������� X��ѵ������
Y = [y1,y2];        %ѵ��Ŀ��1*n����nΪ����������ֵΪ+1��-1��Y��ѵ�������ı�ǩ��+1����������������࣬-1������������ڸ���
svm = svmTrain(X,Y,kertype,C);
plot(svm.Xsv(1,:),svm.Xsv(2,:),'ro'); % ��������߽�
pause;

% ����svm������
%[x1,x2] = meshgrid(-2:0.05:7,-2:0.05:7);  %x1��x2����181*181�ľ���
%[x1,x2] = meshgrid(0:0.05:4.5,1:0.05:5);
[x1,x2] = meshgrid(0:0.02:1,0.4:0.02:1);
[rows,cols] = size(x1);  
nt = rows*cols;                  
Xt = [reshape(x1,1,nt);reshape(x2,1,nt)];
Yt = ones(1,nt);
result = svmTest(svm, Xt, Yt, kertype);
Yd = reshape(result.Y,rows,cols);
contour(x1,x2,Yd,'y');


%ѵ������
trainResult = svmTest(svm,X,Y,kertype);
fprintf('ѵ������Ϊ%f\n',trainResult.accuracy);

%���Ծ���
testResult = svmTest(svm,xTest,yTest,kertype);
fprintf('���Ծ���Ϊ%f\n',testResult.accuracy);



