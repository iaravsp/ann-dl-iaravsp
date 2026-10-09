"""Pré-processamento: ajuste somente no treino."""
NUM = ["age", "avg_glucose_level", "bmi"]

CAT = [
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

TARGET = "stroke"


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import RobustScaler, OneHotEncoder


def build_preprocessor():
    pipe_num = Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("escala", RobustScaler())
    ])
    
    preprocess = ColumnTransformer([
        ("num", pipe_num, NUM),
    
        ("bmi_ausente", MissingIndicator(features="all"), ["bmi"]),
    
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), CAT)
    ])
    
    return preprocess


def balance_training(preprocess, X_train, y_train):
    """Reamostra somente o treino; preprocess deve estar ajustado nesse treino.

    Em validação, chamar apenas após separar a dobra de treino da validação.
    Retorna matriz final, rótulos balanceados e sampler ajustado.
    """
    import numpy as np
    from imblearn.over_sampling import SMOTENC

    entrada = X_train[NUM + CAT].copy()
    entrada[NUM] = preprocess.named_transformers_["num"].transform(X_train[NUM])
    entrada["bmi_ausente"] = X_train["bmi"].isna().astype(int)
    sampler = SMOTENC(
        categorical_features=CAT + ["bmi_ausente"],
        sampling_strategy=1.0, k_neighbors=5, random_state=42
    )
    balanceado, y_balanceado = sampler.fit_resample(entrada, y_train)
    matriz = np.column_stack([
        balanceado[NUM].to_numpy(dtype=float),
        balanceado[["bmi_ausente"]].to_numpy(dtype=float),
        preprocess.named_transformers_["cat"].transform(balanceado[CAT])
    ])
    assert np.isfinite(matriz).all()
    assert balanceado["bmi_ausente"].isin([0, 1]).all()
    inicio = len(NUM) + 1
    for categorias in preprocess.named_transformers_["cat"].categories_:
        fim = inicio + len(categorias)
        bloco = matriz[:, inicio:fim]
        assert np.isin(bloco, [0, 1]).all() and (bloco.sum(axis=1) == 1).all()
        inicio = fim
    return matriz, y_balanceado, sampler
