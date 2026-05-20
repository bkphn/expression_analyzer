def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + FP + FN + TN)

def sensitivity_recall(TP, TN, FP, FN):
    return (TP) / (TP + FN)

def presision(TP, TN, FP, FN):
    return (TP) / (TP + FP)

def f1_score(TP, TN, FP, FN):
    return 0

def specificity(TP, TN, FP, FN):
    return 0

def run_metrics(TP, TN, FP, FN):
    metrics = [accuracy, sensitivity_recall, presision, f1_score, specificity]

    for metric in metrics:
        score = metric(TP, TN, FP, FN)

        print(f"{metric} = {score}")