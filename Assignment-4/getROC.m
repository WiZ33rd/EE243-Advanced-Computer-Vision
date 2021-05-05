function [TPR,FPR] = getROC(pred,gt)

% gt is the ground truth vector of 1 or 0 of size n_samples x 1. 1
% indicates a positive and 0 negative
% pred is a vector of predictions of size n_samples x 1
% TPR is the True Positive Rate
% FPR is the False Positive Rate

% FILL IN
TPR = [];
FPR = [];
for threshold = 0:0.003:1
    lable = pred>threshold;
    %True Positive
    tp_ = lable.*gt;
    TruePositive = sum(tp_);
    %False Positive
    fp_ = lable.*(~gt);
    FalsePositive = sum(fp_);
    %true neigetive
    tn_ = (~lable).*(~gt);
    TrueNegtive = sum(tn_);
    %false negtive
    fn_ = (~lable).*gt;
    FalseNegtive = sum(fn_);
    %TPR is the True Positive Rate
    TPR = [TPR; TruePositive/(TruePositive+FalseNegtive)];
    %FPR is the False Positive Rate
    FPR = [FPR; FalsePositive/(FalsePositive+TrueNegtive)];
end


    
