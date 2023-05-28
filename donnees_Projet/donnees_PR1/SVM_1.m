function [X_VS,w,c,code_retour] = SVM_1(X,Y)
    H = eye(2,2);
    H(3,:) = 0;
    H(:,3) = 0;
    A = zeros(length(X),3);

    for i = 1:length(X)
        A(i,:) = [-Y(i)*X(i,1) -Y(i)*X(i,2) Y(i)];
    end

    [w_tilde,~,code_retour] = quadprog(H,zeros(3,1),A,-ones(1,length(X)));
    c = w_tilde(3);
    w = w_tilde(1:2);

    epsilon = 1e-6;

    bool = Y.*(X*w - c) - 1 <= epsilon;
    X_VS = X(bool,:);
end