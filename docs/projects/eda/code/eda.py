"""Execute da raiz: python docs/projects/eda/code/eda.py."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from preprocessing import build_preprocessor, balance_training
BASE = Path(__file__).resolve().parents[1]
(BASE / "figures").mkdir(exist_ok=True)
def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)


# Célula original 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.dpi"] = 110

df = pd.read_csv(BASE / "data/healthcare-dataset-stroke-data.csv")

display(df.shape)
display(df.dtypes)
display(df.head())

# Célula original 1
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

print("Variáveis numéricas:", len(NUM))
print("Variáveis categóricas:", len(CAT))

# Célula original 2
qualidade = pd.DataFrame({
    "ausentes": df.isna().sum(),
    "ausentes_pct": df.isna().mean() * 100,
    "valores_distintos": df.nunique()
})

display(qualidade.round(2))

print("Linhas duplicadas:", df.duplicated().sum())
print("IDs repetidos:", df["id"].duplicated().sum())
print(
    "Duplicadas sem considerar o ID:",
    df.drop(columns="id").duplicated().sum()
)

# Verificações básicas de domínio, sem decisões de imputação ou escala.
print("Valores negativos:", df[NUM].lt(0).sum().to_dict())
print("Binários fora do domínio:", {
    c: int((~df[c].isin([0, 1])).sum())
    for c in ["hypertension", "heart_disease", "stroke"]
})

# Célula original 3
distribuicao = pd.DataFrame({
    "quantidade": df[TARGET].value_counts(),
    "percentual": df[TARGET].value_counts(normalize=True) * 100
}).sort_index()

display(distribuicao.round(2))

# Célula original 4
from sklearn.model_selection import train_test_split

X = df[NUM + CAT].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Treino:", X_train.shape)
print("Teste:", X_test.shape)

# Célula original 5
#verificar se as proporções das classes foram preservadas
proporcoes = pd.DataFrame({
    "base (%)": y.value_counts(normalize=True) * 100,
    "treino (%)": y_train.value_counts(normalize=True) * 100,
    "teste (%)": y_test.value_counts(normalize=True) * 100
}).sort_index()

display(proporcoes.round(2))

# Célula original 7
display(X_train[NUM].describe().round(2))

# Célula original 9
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

rotulos = {
    "age": "Idade (anos)",
    "avg_glucose_level": "Nível médio de glicose",
    "bmi": "IMC (kg/m²)"
}

for ax, coluna in zip(axes, NUM):
    sns.histplot(data=X_train, x=coluna, bins=25, ax=ax)

    media = X_train[coluna].mean()
    minimo = X_train[coluna].min()
    maximo = X_train[coluna].max()

    ax.axvline(
        media, color="red", linestyle="--",
        label=f"Média: {media:.2f}"
    )
    ax.axvline(
        minimo, color="green", linestyle=":",
        label=f"Mínimo: {minimo:.2f}"
    )
    ax.axvline(
        maximo, color="purple", linestyle=":",
        label=f"Máximo: {maximo:.2f}"
    )

    ax.set_title(rotulos[coluna])
    ax.set_xlabel(rotulos[coluna])
    ax.set_ylabel("Quantidade de registros")
    ax.legend()

fig.suptitle("Figura 1 — Distribuição das variáveis numéricas no treino")
plt.tight_layout()
plt.savefig(BASE / "figures/fig01.png", bbox_inches="tight")
plt.close("all")

# Célula original 10
display(
    X_train[NUM]
    .skew()
    .round(3)
    .rename("Assimetria")
)

# Célula original 11
for coluna in CAT:
    print(f"\n{coluna}")
    print("Quantidade de categorias:", X_train[coluna].nunique())

    frequencias = pd.DataFrame({
        "quantidade": X_train[coluna].value_counts(dropna=False),
        "percentual": (
            X_train[coluna].value_counts(
                dropna=False, normalize=True
            ) * 100
        )
    })

    display(frequencias.round(2))

# Célula original 12
fig, axes = plt.subplots(4, 2, figsize=(13, 16))
axes = axes.flatten()

for ax, coluna in zip(axes, CAT):
    frequencias = X_train[coluna].value_counts(dropna=False)

    ax.barh(
        frequencias.index.astype(str),
        frequencias.values
    )

    ax.invert_yaxis()
    ax.set_title(coluna)
    ax.set_xlabel("Quantidade de registros")
    ax.set_ylabel("Categoria")

    ax.bar_label(ax.containers[0], padding=3)
    ax.margins(x=0.15)

# Remove o painel que sobra
fig.delaxes(axes[-1])

fig.suptitle(
    "Figura 2 — Frequências das variáveis categóricas no treino",
    fontsize=14
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(BASE / "figures/fig02.png", bbox_inches="tight")
plt.close("all")

# Célula original 14
correlacao = X_train[NUM].corr(method="spearman")

display(correlacao.round(3))

plt.figure(figsize=(6, 4))

sns.heatmap(
    correlacao,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0
)

plt.title("Figura 3 — Correlação de Spearman no treino")
plt.tight_layout()
plt.savefig(BASE / "figures/fig03.png", bbox_inches="tight")
plt.close("all")

# Célula original 15
pares = [
    ("age", "avg_glucose_level"),
    ("age", "bmi"),
    ("avg_glucose_level", "bmi")
]

rotulos = {
    "age": "Idade (anos)",
    "avg_glucose_level": "Nível médio de glicose",
    "bmi": "IMC (kg/m²)"
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (x, y) in zip(axes, pares):
    sns.scatterplot(
        data=X_train,
        x=x,
        y=y,
        alpha=0.25,
        s=15,
        ax=ax
    )

    ax.set_xlabel(rotulos[x])
    ax.set_ylabel(rotulos[y])
    ax.set_title(f"Spearman: {correlacao.loc[x, y]:.3f}")

fig.suptitle("Figura 4 — Relações entre as variáveis numéricas no treino")
plt.tight_layout()
plt.savefig(BASE / "figures/fig04.png", bbox_inches="tight")
plt.close("all")

# Célula original 17
df_train = X_train.copy()
df_train[TARGET] = y_train

for coluna in CAT:
    resumo = df_train.groupby(coluna)[TARGET].agg(
        total="count",
        casos_avc="sum",
        proporcao_avc_pct="mean"
    )

    resumo["proporcao_avc_pct"] *= 100

    print(f"\n{coluna}")
    display(resumo.round(2))

# Célula original 18
selecionadas = ["hypertension", "heart_disease", "smoking_status"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
proporcao_geral = y_train.mean() * 100

for ax, coluna in zip(axes, selecionadas):
    resumo = df_train.groupby(coluna)[TARGET].agg(
        total="count",
        proporcao="mean"
    )

    resumo["proporcao"] *= 100

    barras = ax.bar(
        resumo.index.astype(str),
        resumo["proporcao"]
    )

    ax.bar_label(
        barras,
        labels=[
            f"{p:.2f}%\n(n={n})"
            for p, n in zip(resumo["proporcao"], resumo["total"])
        ],
        padding=3
    )

    ax.axhline(
        proporcao_geral,
        color="red",
        linestyle="--",
        label=f"Treino: {proporcao_geral:.2f}%"
    )

    ax.set_title(coluna)
    ax.set_xlabel("Categoria")
    ax.set_ylabel("Registros com AVC (%)")
    ax.set_ylim(0, 20)
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend()

fig.suptitle("Figura 5 — Proporção de AVC por categoria no treino")
plt.tight_layout()
plt.savefig(BASE / "figures/fig05.png", bbox_inches="tight")
plt.close("all")

# Tabela 6A: composição etária, somente no treino original.
for coluna in ["ever_married", "work_type", "smoking_status"]:
    print("Idade por categoria:", coluna)
    display(df_train.groupby(coluna).agg(
        n=("age", "size"), idade_mediana=("age", "median"),
        avc_pct=(TARGET, lambda valores: valores.mean() * 100)
    ).round(2))

# Célula original 20
display(
    df_train.groupby(TARGET)[NUM]
    .agg(["count", "median", "mean", "std"])
    .round(2)
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, coluna in zip(axes, NUM):
    sns.boxplot(
        data=df_train,
        x=TARGET,
        y=coluna,
        order=[0, 1],
        ax=ax
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Sem AVC", "Com AVC"])
    ax.set_xlabel("")
    ax.set_ylabel(rotulos[coluna])
    ax.set_title(rotulos[coluna])

fig.suptitle("Figura 6 — Variáveis numéricas por classe de AVC no treino")
plt.tight_layout()
plt.savefig(BASE / "figures/fig06.png", bbox_inches="tight")
plt.close("all")

# Célula original 21
display(
    df_train.loc[
        (df_train[TARGET] == 1) & (df_train["age"] < 18)
    ]
)

# Célula original 24
q1 = X_train[NUM].quantile(0.25)
q3 = X_train[NUM].quantile(0.75)
iqr = q3 - q1

limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

extremos = (
    X_train[NUM].lt(limite_inferior)
    | X_train[NUM].gt(limite_superior)
)

resumo_extremos = pd.DataFrame({
    "limite_inferior": limite_inferior,
    "limite_superior": limite_superior,
    "quantidade": extremos.sum(),
    "pct_dos_preenchidos": (
        extremos.sum() / X_train[NUM].count() * 100
    )
})

display(resumo_extremos.round(2))

print(
    "Registros com extremo em pelo menos uma variável:",
    extremos.any(axis=1).sum()
)

# Célula original 25
for coluna in ["avg_glucose_level", "bmi"]:
    situacao = pd.Series("Dentro dos limites", index=X_train.index)

    situacao.loc[extremos[coluna]] = "Extremo pelo IQR"
    situacao.loc[X_train[coluna].isna()] = "Ausente"

    resumo = pd.crosstab(situacao, y_train).reindex(
        columns=[0, 1], fill_value=0
    )

    resumo.columns = ["Sem AVC", "Com AVC"]
    resumo["Total"] = resumo["Sem AVC"] + resumo["Com AVC"]
    resumo["AVC (%)"] = resumo["Com AVC"] / resumo["Total"] * 100

    print(f"\n{coluna}")
    display(resumo.round(2))

# Célula original 26
display(
    df_train.nlargest(10, "bmi")[
        ["age", "bmi", "avg_glucose_level", TARGET]
    ]
)

# Célula original 27
preprocess = build_preprocessor()
X_train_t = preprocess.fit_transform(X_train)
X_test_t = preprocess.transform(X_test)

# Célula original 28
print("Treino após pipeline:", X_train_t.shape)
print("Teste após pipeline:", X_test_t.shape)

print("NaN no treino:", np.isnan(X_train_t).sum())
print("NaN no teste:", np.isnan(X_test_t).sum())

display(preprocess.get_feature_names_out())

print(
    "Medianas aprendidas:",
    preprocess.named_transformers_["num"]
    .named_steps["imputacao"].statistics_
)

# Balanceamento: saída adicional; projeções continuam no treino original.
X_train_balanceado_t, y_train_balanceado, smote = balance_training(
    preprocess, X_train, y_train
)
print("Treino balanceado:", X_train_balanceado_t.shape)
print("NaN balanceado:", np.isnan(X_train_balanceado_t).sum())
display(y_train_balanceado.value_counts().sort_index())
assert X_train_balanceado_t.shape == (7778, 24)
assert y_train_balanceado.value_counts().to_dict() == {0: 3889, 1: 3889}
for coluna in CAT:
    print("Categorias inéditas no teste:", coluna,
          set(X_test[coluna]) - set(X_train[coluna]))

# Célula original 29
from sklearn.decomposition import PCA

pca = PCA(svd_solver="full")
pca.fit(X_train_t)

variancia = pd.DataFrame({
    "componente": [
        f"PC{i + 1}"
        for i in range(len(pca.explained_variance_ratio_))
    ],
    "variancia_pct": pca.explained_variance_ratio_ * 100,
    "acumulada_pct": np.cumsum(
        pca.explained_variance_ratio_
    ) * 100
})

display(variancia.round(2))

print(
    "Variância explicada por PC1 + PC2:",
    f"{variancia['variancia_pct'].iloc[:2].sum():.2f}%"
)

# Figura 7: contribuição individual e soma acumulada no treino original.
componentes = np.arange(1, len(variancia) + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
cores = ["#176B87" if i < 2 else "#B8CCD4" for i in range(len(variancia))]
axes[0].bar(componentes, variancia["variancia_pct"], color=cores,
            label="Variância de cada componente")
for i in range(2):
    axes[0].text(i + 1, variancia.loc[i, "variancia_pct"] + .7,
                 f"{variancia.loc[i, 'variancia_pct']:.2f}%", ha="center", fontsize=9)
axes[0].set(title="Quanto cada componente acrescenta?",
            xlabel="Componente principal (em ordem)", ylabel="Variância individual (%)",
            ylim=(0, 35))
axes[0].legend(frameon=False, fontsize=8)
axes[1].plot(componentes, variancia["acumulada_pct"], color="#176B87",
             marker="o", markersize=3, label="Soma das contribuições")
axes[1].axhline(80, color="#A2ADB4", ls="--", lw=1, label="Referências: 80% e 95%")
axes[1].axhline(95, color="#A2ADB4", ls="--", lw=1)
for k, offset in [(2, (25, -18)), (7, (18, -35)), (12, (20, -24))]:
    valor = variancia.loc[k-1, "acumulada_pct"]
    axes[1].scatter(k, valor, color="#D87524", s=38, zorder=4)
    axes[1].annotate(f"{k} componentes: {valor:.2f}%", (k, valor),
                     xytext=offset, textcoords="offset points", fontsize=9,
                     arrowprops=dict(arrowstyle="-", color="#D87524"))
axes[1].set(title="Quanto acumulamos ao manter mais componentes?",
            xlabel="Número de componentes mantidos", ylabel="Variância acumulada (%)",
            ylim=(0, 108))
axes[1].legend(loc="lower right", frameon=False, fontsize=8)
for ax in axes:
    ax.set_xticks([1, 2, 4, 7, 10, 12, 16, 20, 24])
    ax.set_xlim(.3, 24.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.16)
    ax.set_axisbelow(True)
fig.suptitle("Figura 7 — Variância explicada pelo PCA no treino original")
fig.tight_layout()
fig.savefig(BASE / "figures/fig07.png", dpi=160, bbox_inches="tight")
plt.close("all")

# Célula original 31
pesos = pd.DataFrame(
    pca.components_[:2].T,
    index=preprocess.get_feature_names_out(),
    columns=["PC1", "PC2"]
)

for componente in ["PC1", "PC2"]:
    principais = (
        pesos[componente]
        .abs()
        .sort_values(ascending=False)
        .head(8)
        .index
    )

    print(f"\nMaiores pesos de {componente}")
    display(pesos.loc[principais, [componente]].round(3))

# Célula original 32
projecao = pca.transform(X_train_t)[:, :2]

fig, ax = plt.subplots(figsize=(8, 5))

for classe, nome, cor in [
    (0, "Sem AVC", "steelblue"),
    (1, "Com AVC", "darkorange")
]:
    mascara = y_train.to_numpy() == classe

    ax.scatter(
        projecao[mascara, 0],
        projecao[mascara, 1],
        label=nome,
        color=cor,
        alpha=0.35 if classe == 0 else 0.85,
        s=15 if classe == 0 else 30
    )

ax.set_xlabel(f"PC1 ({variancia.loc[0, 'variancia_pct']:.2f}%)")
ax.set_ylabel(f"PC2 ({variancia.loc[1, 'variancia_pct']:.2f}%)")
ax.set_title("Figura 8 — PCA do treino, colorido por classe de AVC")
ax.legend()
plt.tight_layout()
plt.savefig(BASE / "figures/fig08.png", bbox_inches="tight")
plt.close("all")

# Célula original 33
from sklearn.manifold import TSNE

perplexidades = [10, 50]
projecoes_tsne = {}
classes = y_train.to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, perplexidade in zip(axes, perplexidades):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexidade,
        init="pca",
        learning_rate="auto",
        random_state=42
    )

    projecao = tsne.fit_transform(X_train_t)
    projecoes_tsne[perplexidade] = projecao

    for classe, cor, legenda in [
        (0, "steelblue", "Sem AVC"),
        (1, "darkorange", "Com AVC")
    ]:
        mascara = classes == classe

        ax.scatter(
            projecao[mascara, 0],
            projecao[mascara, 1],
            color=cor,
            label=legenda,
            alpha=0.35 if classe == 0 else 0.85,
            s=15 if classe == 0 else 30
        )

    ax.set_title(f"Perplexity = {perplexidade}")
    ax.set_xlabel("Dimensão 1 do t-SNE")
    ax.set_ylabel("Dimensão 2 do t-SNE")
    ax.legend()

fig.suptitle("Figura 9 — t-SNE do treino, colorido por classe de AVC")
plt.tight_layout()
plt.savefig(BASE / "figures/fig09.png", bbox_inches="tight")
plt.close("all")

# Célula original 35
from umap import UMAP

vizinhos = [10, 50]
projecoes_umap = {}
classes = y_train.to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, n in zip(axes, vizinhos):
    umap = UMAP(
        n_components=2,
        n_neighbors=n,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        n_jobs=1
    )

    projecao = umap.fit_transform(X_train_t)
    projecoes_umap[n] = projecao

    for classe, cor, legenda in [
        (0, "steelblue", "Sem AVC"),
        (1, "darkorange", "Com AVC")
    ]:
        mascara = classes == classe

        ax.scatter(
            projecao[mascara, 0],
            projecao[mascara, 1],
            color=cor,
            label=legenda,
            alpha=0.35 if classe == 0 else 0.85,
            s=15 if classe == 0 else 30
        )

    ax.set_title(f"n_neighbors = {n}")
    ax.set_xlabel("Dimensão 1 do UMAP")
    ax.set_ylabel("Dimensão 2 do UMAP")
    ax.legend()

fig.suptitle("Figura 10 — UMAP do treino, colorido por classe de AVC")
plt.tight_layout()
plt.savefig(BASE / "figures/fig10.png", bbox_inches="tight")
plt.close("all")

# Verificação das matrizes originais, preservadas após a reamostragem.
assert X_train_t.shape == (4088, 24)
assert X_test_t.shape == (1022, 24)
assert np.isfinite(X_train_t).all() and np.isfinite(X_test_t).all()
print("Versões executadas:")
from importlib.metadata import version
for pacote in ["numpy", "pandas", "matplotlib", "seaborn", "scikit-learn", "imbalanced-learn", "umap-learn"]:
    print(pacote, version(pacote))
