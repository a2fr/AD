function [X_VS,Y_VS,Alpha_VS,c,code_retour] = SVM_3_souple(X,Y,sigma,lambda)
    n = length(X);

    K = zeros(n,n);
    for i = 1:n
        for j = 1:n
            K(i,j) = exp(-norm(X(i)-X(j)) / (2*sigma^2));
        end
    end

    epsilon = 1e-6;

    H = diag(Y')*K*diag(Y);

    f = -ones(n,1);
    Aeq = Y';
    beq = 0;
    lb = zeros(n,1);
    ub = lambda*ones(n,1);

    [alpha,~,code_retour] = quadprog(H,f,[],[],Aeq,beq,lb,ub);    

    idx = alpha>epsilon;
    lidx = idx & (alpha<lambda);
    Y_VS = Y(lidx);
    Alpha_VS = alpha(lidx);

    c = sum(Alpha_VS.*Y_VS.*K(lidx,1)) - 1/Y_VS(1);
    X_VS = X(idx,:);

end