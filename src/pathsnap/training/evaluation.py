from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix


def classification_metrics(y_true, y_pred, probabilities=None):
    result={"accuracy":accuracy_score(y_true,y_pred),"precision":precision_score(y_true,y_pred,zero_division=0),"recall":recall_score(y_true,y_pred,zero_division=0),"macro_f1":f1_score(y_true,y_pred,average="macro",zero_division=0),"confusion_matrix":confusion_matrix(y_true,y_pred).tolist()}
    if probabilities is not None: result["log_loss"]=log_loss(y_true,probabilities,labels=[0,1])
    return result
