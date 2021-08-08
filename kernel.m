%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function K = kernel(X,Y,type)
%X 维数*个数
switch type
case 'linear' % 线性核
   %K = X+Y;   %(bug1)
   K = X'*Y;
case 'rbf' % 高斯核（非线性）
    delta = 0.2;
    delta = delta*delta;
    XX = sum(X'.*X',2);
    YY = sum(Y'.*Y',2);
    XY = X'*Y;
    K = abs(repmat(XX,[1 size(YY,1)]) + repmat(YY',[size(XX,1) 1]) - 2*XY);
    K = exp(-K./delta);
end
