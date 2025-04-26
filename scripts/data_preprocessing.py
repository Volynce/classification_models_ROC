import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(df):
    # Удаление первого столбца
    df = df.drop(df.columns[0], axis=1)

    # One Hot Encoding для текстовых столбцов
    df = pd.get_dummies(df, columns=['color', 'clarity'])

    # Преобразование целевой переменной
    y = df['cut']
    X = df.drop('cut', axis=1)
    y = y.replace({'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4})

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)

    return X_train, X_test, y_train, y_test
