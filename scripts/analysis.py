from data_preprocessing import preprocess_data


def run_analysis(X_train, X_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    # Обучение моделей
    dtc = DecisionTreeClassifier(random_state=51)
    dtc.fit(X_train, y_train)

    rfc = RandomForestClassifier(random_state=51)
    rfc.fit(X_train, y_train)

    knc = KNeighborsClassifier()
    knc.fit(X_train, y_train)

    return dtc, rfc, knc


if __name__ == "__main__":
    # Здесь ты должен вызвать функцию preprocess_data, чтобы получить обучающие данные
    from data_loader import load_data

    df = load_data('diamonds.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Теперь можешь вызвать run_analysis с полученными данными
    dtc, rfc, knc = run_analysis(X_train, X_test, y_train, y_test)
