import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names):
    # Важность признаков
    importance = model.feature_importances_
    plt.barh(feature_names, importance)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance from Random Forest')
    plt.show()
