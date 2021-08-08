%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result = svmTest(svm, Xt, Yt, kertype) % svm记录了支持向量SV的alpha值、样本值和分类标签，可以通过svm计算分类器；Xt是测试样本，Yt是测试样本的标签
temp = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,svm.Xsv,kertype);
total_b = svm.Ysv-temp;
b = mean(total_b); % 计算y=wx+b公式中的 b值
w = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,Xt,kertype); % 计算y=wx+b公式中的 w值
result.score = w + b;
Y = sign(w+b);
result.Y = Y;
result.accuracy = size(find(Y==Yt))/size(Yt);  % 预测精度
