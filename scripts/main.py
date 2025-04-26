import os
from data_loader import load_data
from data_preprocessing import preprocess_data
from analysis import run_analysis
from roc_curve import plot_roc_multiclass
from visualization import plot_feature_importance

def main():
    # Загружаем данные
    print("Загрузка данных...")
    df = load_data('diamonds.csv')

    # Предобработка данных
    print("Предобработка данных...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Анализ данных и обучение моделей
    print("Анализ данных и обучение моделей...")
    dtc, rfc, knc = run_analysis(X_train, X_test, y_train, y_test)

    # Построение ROC кривой для каждой модели
    print("Построение ROC кривых для каждой модели...")
    plot_roc_multiclass(y_test, dtc, X_test, filename='roc_curve_dtc.png')
    plot_roc_multiclass(y_test, rfc, X_test, filename='roc_curve_rfc.png')
    plot_roc_multiclass(y_test, knc, X_test, filename='roc_curve_knc.png')

    # Визуализация важности признаков (для случайного леса)
    print("Визуализация важности признаков...")
    plot_feature_importance(rfc, X_train.columns)

    print("Процесс завершен!")

if __name__ == "__main__":
    main()
