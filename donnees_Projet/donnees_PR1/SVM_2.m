function [X_VS,w,c,code_retour] = SVM_2(X,Y)
    Aeq = Y';
    n = length(X);
    H = diag(Y')*(X*X')*diag(Y);

    f = -ones(n,1);

    [alpha,~,code_retour] = quadprog(H,f,-eye(n),zeros(n,1),Aeq,0);
    
    epsilon = 1e-6;
    idx = alpha >= epsilon;
    X_VS = X(idx,:);
    Y_VS = Y(idx);
    alpha_VS = alpha(idx);

    w = (alpha_VS.*Y_VS)'*X_VS;
    c = X_VS(1,:)*w' - 1/Y(1);

    w = w';
end