%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function svm = svmTrain(X,Y,kertype,C) % X��ѵ��������Y��ѵ�������ı�ǩ

options = optimset;    % Options�����������㷨��ѡ�����������
options.LargeScale = 'off';
options.Display = 'off';

n = length(Y);
H = kernel(X,X,kertype);  %(bug2)
H = H.*(Y'*Y);
%f = zeros(n,1); %fΪ1*n��-1,f�൱��Quadprog�����е�c %(bug3)
f = -ones(n,1);
A = [];
b = [];
%Aeq = 0; %�൱��Quadprog�����е�A1,b1 %(bug4)
Aeq = Y;
beq = 0;
%lb = ones(n,1); %�൱��Quadprog�����е�LB��%(bug5) 
lb = zeros(n,1);
%ub = ones(n,1);  %(bug6)
ub = C*ones(n,1);
a0 = zeros(n,1);  % a0�ǽ�ĳ�ʼ����ֵ
[a,fval,eXitflag,output,lambda]  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options); % ����matlab�Դ������Quadprog����ĺ���

epsilon = 1e-8;                     
sv_label = find(abs(a)>epsilon);  %0<a<a(max)����ΪxΪ֧��������ֻ�е�alpha����epsilon���ű��ж�Ϊ֧������support vector��SV��     
svm.a = a(sv_label); % ֧������SV��alphaֵ
svm.Xsv = X(:,sv_label); % ֧������SV������ֵ
svm.Ysv = Y(sv_label); % ֧��������SV�����ǩ
svm.svnum = length(sv_label);
%svm.label = sv_label;