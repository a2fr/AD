function [X_VS,Y_VS,Alpha_VS,c,code_retour] = SVM_3(X,Y,sigma)
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

    [alpha,~,code_retour] = quadprog(H,f,[],[],Aeq,beq,lb,[]);

    idx = alpha > epsilon;
    X_VS = X(idx,:);
    Y_VS = Y(idx);
    Alpha_VS = alpha(idx);

    c = sum(Alpha_VS.*Y_VS.*K(idx,1)) - 1/Y_VS(1);

end