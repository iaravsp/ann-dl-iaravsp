"""Exercise 3 - Preparing the Spaceship Titanic data for a tanh network.

O dataset nao e gerado pelo script: baixe train.csv da competicao
https://www.kaggle.com/competitions/spaceship-titanic e deixe em
data/spaceship-titanic/train.csv.

Run from the repository root:

    python docs/exercises/data/code/ex3_preprocessing.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA = REPO_ROOT / "data" / "spaceship-titanic" / "train.csv"

SEED = 42
TEST_SIZE = 0.2
TARGET = "Transported"

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERIC = ["Age"] + SPEND
CATEGORICAL = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Cabin"]
IDENTIFIERS = ["PassengerId", "Name"]


def load():
    if not DATA.exists():
        raise SystemExit(f"dataset nao encontrado em {DATA}")
    return pd.read_csv(DATA)


def target_balance(df):
    """Proporcao de cada classe do alvo."""
    return df[TARGET].value_counts(normalize=True).sort_index()


def missing_table(df):
    """Contagem absoluta e percentual de faltantes, so das colunas afetadas."""
    count = df.isna().sum()
    table = pd.DataFrame({"faltantes": count, "percentual": 100 * count / len(df)})
    return table[table["faltantes"] > 0].sort_values("faltantes", ascending=False)


def spend_stats(df):
    """Media, mediana e maximo das colunas de gasto."""
    return df[SPEND].agg(["mean", "median", "max"]).T


def split(df):
    """Split estratificado 80/20 com semente fixa, antes de qualquer transformacao.

    Estratificar mantem a proporcao de Transported igual nas duas partes.
    O split vem primeiro porque imputacao e escalonamento aprendem parametros
    (mediana, media, desvio) que so podem ser estimados no treino.
    """
    return train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df[TARGET],
    )


def main():
    df = load()

    print("=== A - conhecendo os dados ===")
    print(f"shape: {df.shape}")
    print(f"\nbalanceamento de {TARGET}:")
    print(target_balance(df).to_string())

    print("\ncolunas numericas:", NUMERIC)
    print("colunas categoricas:", CATEGORICAL)
    print("identificadores (serao descartados):", IDENTIFIERS)

    print("\nvalores faltantes:")
    print(missing_table(df).round(2).to_string())

    print("\nestatisticas das colunas de gasto (dataset completo):")
    print(spend_stats(df).round(2).to_string())

    print("\n=== B - split antes de transformar ===")
    train_df, test_df = split(df)
    print(f"treino: {train_df.shape}  teste: {test_df.shape}")
    print(f"proporcao de {TARGET} no treino: {train_df[TARGET].mean():.4f}")
    print(f"proporcao de {TARGET} no teste:  {test_df[TARGET].mean():.4f}")

    print("\nFoodCourt no treino, antes de qualquer transformacao:")
    print(f"  media:   {train_df['FoodCourt'].mean():.2f}")
    print(f"  mediana: {train_df['FoodCourt'].median():.2f}")


if __name__ == "__main__":
    main()
