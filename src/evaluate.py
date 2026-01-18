from sklearn.metrics import accuracy_score, confusion_matrix

def get_accuracy(y_test, y_pred):
    return accuracy_score(y_test, y_pred)

def get_confusion_matrix(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)