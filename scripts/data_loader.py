
import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Пример использования
if __name__ == "__main__":
    df = load_data('diamonds.csv')
    print(df.info())
