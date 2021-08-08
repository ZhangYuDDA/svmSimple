%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result = svmTest(svm, Xt, Yt, kertype) % svm��¼��֧������SV��alphaֵ������ֵ�ͷ����ǩ������ͨ��svm�����������Xt�ǲ���������Yt�ǲ��������ı�ǩ
temp = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,svm.Xsv,kertype);
total_b = svm.Ysv-temp;
b = mean(total_b); % ����y=wx+b��ʽ�е� bֵ
w = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,Xt,kertype); % ����y=wx+b��ʽ�е� wֵ
result.score = w + b;
Y = sign(w+b);
result.Y = Y;
result.accuracy = size(find(Y==Yt))/size(Yt);  % Ԥ�⾫��
