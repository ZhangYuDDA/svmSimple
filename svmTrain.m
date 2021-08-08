%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function svm = svmTrain(X,Y,kertype,C) % X是训练样本，Y是训练样本的标签

options = optimset;    % Options是用来控制算法的选项参数的向量
options.LargeScale = 'off';
options.Display = 'off';

n = length(Y);
H = kernel(X,X,kertype);  %(bug2)
H = H.*(Y'*Y);
%f = zeros(n,1); %f为1*n个-1,f相当于Quadprog函数中的c %(bug3)
f = -ones(n,1);
A = [];
b = [];
%Aeq = 0; %相当于Quadprog函数中的A1,b1 %(bug4)
Aeq = Y;
beq = 0;
%lb = ones(n,1); %相当于Quadprog函数中的LB，%(bug5) 
lb = zeros(n,1);
%ub = ones(n,1);  %(bug6)
ub = C*ones(n,1);
a0 = zeros(n,1);  % a0是解的初始近似值
[a,fval,eXitflag,output,lambda]  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options); % 调用matlab自带的求解Quadprog问题的函数

epsilon = 1e-8;                     
sv_label = find(abs(a)>epsilon);  %0<a<a(max)则认为x为支持向量，只有当alpha大于epsilon，才被判断为支持向量support vector（SV）     
svm.a = a(sv_label); % 支持向量SV的alpha值
svm.Xsv = X(:,sv_label); % 支持向量SV的样本值
svm.Ysv = Y(sv_label); % 支持向量的SV分类标签
svm.svnum = length(sv_label);
%svm.label = sv_label;