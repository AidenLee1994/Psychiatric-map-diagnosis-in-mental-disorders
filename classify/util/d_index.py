import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix


def compute_measure(predicted_label, true_label):
    cnf_matrix = confusion_matrix(true_label, predicted_label)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    tp_fn=(TP + FN)

    TPR = TP / tp_fn
    # Specificity or true negative rate
    tn_fp=(TN + FP)

    TNR = TN / tn_fp
    # Precision or positive predictive value
    tp_fp=(TP + FP)

    PPV = TP / tp_fp
    # Negative predictive value
    tn_fn=(TN + FN)

    NPV = TN / tn_fn
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # F1
    F_1 = 2 * (PPV * TPR) / (PPV + TPR)
    # Overall accuracy for each class
    ACC_Class = (TP + TN) / (TP + FP + FN + TN)
    # Average accuracy
    ACC = np.sum(np.diag(cnf_matrix)) / cnf_matrix.sum()

    d_idx = np.log2(1 + ACC) + np.log2(1 + (TPR + TNR) / 2)

    ans = []
    ans.append(ACC)
    ans.append(d_idx.mean())
    ans.append(TPR.mean())
    ans.append(TNR.mean())
    ans.append(PPV.mean())
    ans.append(NPV.mean())

    return ans




def multi_compute_measure(class_num, predicted_label, true_label):
    acc_list = []
    d_idx_list = []
    sen_list = []
    spc_list = []
    ppr_list = []
    npr_list = []
    for class_name in range(0,class_num):
        t_idx = (predicted_label == true_label)  # truely predicted
        f_idx = np.logical_not(t_idx)

        p_idx = (class_name == true_label)
        n_idx = np.logical_not(p_idx)

        tp = np.sum(np.logical_and(t_idx, p_idx))  # TP
        tn = np.sum(np.logical_and(t_idx, n_idx))  # TN

        fp = np.sum(n_idx) - tn
        fn = np.sum(p_idx) - tp

        tp_fp_tn_fn_list = []
        tp_fp_tn_fn_list.append(tp)
        tp_fp_tn_fn_list.append(fp)
        tp_fp_tn_fn_list.append(tn)
        tp_fp_tn_fn_list.append(fn)
        tp_fp_tn_fn_list = np.array(tp_fp_tn_fn_list)

        tp = tp_fp_tn_fn_list[0]
        fp = tp_fp_tn_fn_list[1]
        tn = tp_fp_tn_fn_list[2]
        fn = tp_fp_tn_fn_list[3]

        with np.errstate(divide='ignore'):
            tp_fn=(tp + fn)
            if tp_fn==0:
                tp_fn=10
            sen = (1.0 * tp) / tp_fn
        with np.errstate(divide='ignore'):
            tn_fp=(tn + fp)
            if tn_fp==0:
                tn_fp=10
            spc = (1.0 * tn) / tn_fp
        with np.errstate(divide='ignore'):
            tp_fp=(tp + fp)
            if tp_fp==0:
                tp_fp=10
            ppr = (1.0 * tp) / tp_fp
        with np.errstate(divide='ignore'):
            tn_fn=(tn + fn)
            if tn_fn==0:
                tn_fn=10
            npr = (1.0 * tn) / tn_fn

        acc = (tp + tn) * 1.0 / (tp + fp + tn + fn)
        d_idx = np.log2(1 + acc) + np.log2(1 + (sen + spc) / 2)
        acc_list.append(acc)
        d_idx_list.append(d_idx)
        sen_list.append(sen)
        spc_list.append(spc)
        ppr_list.append(ppr)
        npr_list.append(npr)

    ans = []
    ans.append(np.mean(acc_list))
    ans.append(np.mean(d_idx_list))
    ans.append(np.mean(sen_list))
    ans.append(np.mean(spc_list))
    ans.append(np.mean(ppr_list))
    ans.append(np.mean(npr_list))

    return ans

# a=np.array([1,1,0,1])
# b=np.array([0,1,1,1])
# compute_measure(a,b)