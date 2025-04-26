import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_roc_multiclass(y_true, model, X_test, filename='roc_curve.png'):
    # Преобразуем метки в бинарный формат для каждого класса
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])  # Предположим, у нас 5 классов
    n_classes = y_true_bin.shape[1]

    # Получаем вероятности для каждого класса
    y_pred_prob = model.predict_proba(X_test)

    # Создаем график
    plt.figure(figsize=(10, 8))

    # Строим ROC кривую для каждого класса
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve class {i} (area = {roc_auc:0.2f})')

    # Диагональная линия (случайная модель)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Multiclass')
    plt.legend(loc='lower right')

    # Сохраняем график
    plt.savefig(filename, format='png')
    plt.close()
