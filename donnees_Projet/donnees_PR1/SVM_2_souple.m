function [X_VS,w,c,code_retour] = SVM_2_souple(X,Y,lambda)
    Aeq = Y';
    beq = 0;
    n = length(X);
    H = diag(Y)*(X*X')*diag(Y);

    f = -ones(n,1);

    lb = zeros(n,1);
    ub = lambda*ones(n,1);

    [alpha,~,code_retour] = quadprog(H,f,[],[],Aeq,beq,lb,ub);
    
    epsilon = 1e-6;
    idx = alpha > epsilon;
    X_VS = X(idx,:);
    Y_VS = Y(idx);
    alpha_VS = alpha(idx,:);

    w = X_VS'*diag(Y_VS)*alpha_VS;

    lidx = idx & (alpha<lambda);
    X_VS = X(lidx,:);
    Y_VS = Y(lidx);

    c = X_VS(1,:)*w - 1/Y_VS(1);
    X_VS = X(idx,:);

end