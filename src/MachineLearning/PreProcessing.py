# tiền xử lý dữ liệu và xử lý mất cân bằng
def _convert_format(data):
    import pandas as pd
    data.replace(',', '.', regex=True, inplace=True)

    # Convert all object columns to numeric, where possible
    for col in data.select_dtypes(include='object').columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    print('finished converting data to correct format')
    # Display the updated dataset info
    return data


def _normalization(data):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    print('finished normalization')

    return data


def _handle_missing_value(data):
    data.fillna(data.median(), inplace=True)
    print('finished filling missing values')

    return data


def _handle_class_imbalance(data):
    from imblearn.over_sampling import ADASYN
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    ada = ADASYN(random_state=42)
    X_resampled, y_resampled = ada.fit_resample(X, y)
    print('finished balancing data')

    return X_resampled, y_resampled


def pre_processing(data, pytorch=None):
    import torch
    data = _convert_format(data)
    data = _normalization(data)
    data = _handle_missing_value(data)
    X_resampled, y_resampled = _handle_class_imbalance(data)
    print('finished pre-processing data')

    if pytorch:
        X_tensor = torch.tensor(X_resampled.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_resampled.values, dtype=torch.float32)
        return X_tensor, y_tensor

    return X_resampled, y_resampled
