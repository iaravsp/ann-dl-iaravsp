from pathlib import Path
import matplotlib
matplotlib.use("Agg")
FIGURAS = Path(__file__).resolve().parents[1] / "figures"
FIGURAS.mkdir(exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

covariancia = [[0.5, 0],
              [0, 0.5]]

# Cria 1000 pontos de cada classe
classe_0 = rng.multivariate_normal(
    mean=[1.5, 1.5],
    cov=covariancia,
    size=1000
)

classe_1 = rng.multivariate_normal(
    mean=[5, 5],
    cov=covariancia,
    size=1000
)

plt.figure(figsize=(7, 5))

plt.scatter(
    classe_0[:, 0], classe_0[:, 1],
    label="Classe 0", alpha=0.5, s=15
)

plt.scatter(
    classe_1[:, 0], classe_1[:, 1],
    label="Classe 1", alpha=0.5, s=15
)

plt.xlabel("x₁")
plt.ylabel("x₂")
plt.title("Dados do exercício 1")
plt.legend()
plt.grid(alpha=0.2)
plt.savefig(FIGURAS / "fig1_dados.png", dpi=160, bbox_inches="tight")
plt.close()

X = np.vstack((classe_0, classe_1))

y = np.concatenate((
    np.zeros(1000),  # Classe 0
    np.ones(1000)    # Classe 1
))

print("Formato de X:", X.shape)
print("Formato de y:", y.shape)
print("Primeiro ponto:", X[0])
print("Classe do primeiro ponto:", y[0])

def prever(x, w, b):
    z = np.dot(w, x) + b

    if z >= 0:
        return 1
    else:
        return 0

def treinar_perceptron(X, y, w_inicial, ordens, taxa=0.01):
    w = w_inicial.copy()
    b = 0.0

    historico = []
    historico_atualizacoes = []

    for indices in ordens:
        atualizacoes = 0

        for i in indices:
            previsao = prever(X[i], w, b)
            erro = y[i] - previsao

            if erro != 0:
                w = w + taxa * erro * X[i]
                b = b + taxa * erro
                atualizacoes += 1

        acertos = 0

        for i in range(len(X)):
            if prever(X[i], w, b) == y[i]:
                acertos += 1

        historico.append(acertos / len(X))
        historico_atualizacoes.append(atualizacoes)

        if atualizacoes == 0:
            break

    return {
        "pesos": w.copy(),
        "bias": b,
        "acuracia": historico[-1],
        "epocas": len(historico),
        "historico": historico,
        "atualizacoes": historico_atualizacoes
    }

w_inicial = rng.normal(0, 0.01, size=2)

print("\nPesos iniciais:", w_inicial)
print("Norma dos pesos iniciais:", np.linalg.norm(w_inicial))

ordens = [
    rng.permutation(len(X))
    for _ in range(100)
]

resultado_1 = treinar_perceptron(
    X, y, w_inicial, ordens, taxa=0.01
)

resultado_taxa_1 = treinar_perceptron(
    X, y, w_inicial, ordens, taxa=1.0
)

for taxa, resultado in [
    (0.01, resultado_1),
    (1.0, resultado_taxa_1)
]:
    pesos = resultado["pesos"]
    direcao = pesos / np.linalg.norm(pesos)

    print(f"\nTaxa: {taxa}")
    print("Pesos:", pesos)
    print("Bias:", resultado["bias"])
    print("Épocas:", resultado["epocas"])
    print(f"Acurácia: {resultado['acuracia']:.2%}")
    print("Direção de w:", direcao)
    print("Norma de w:", np.linalg.norm(pesos))
    print("Atualizações por época:", resultado["atualizacoes"])

# Ângulo entre as duas direções, para comparar as fronteiras
direcao_001 = resultado_1["pesos"] / np.linalg.norm(resultado_1["pesos"])
direcao_1 = resultado_taxa_1["pesos"] / np.linalg.norm(resultado_taxa_1["pesos"])
cosseno = np.clip(np.dot(direcao_001, direcao_1), -1, 1)

print("\nÂngulo entre as direções (graus):", np.degrees(np.arccos(cosseno)))

# Mantém os gráficos existentes usando o resultado da taxa 0.01
w = resultado_1["pesos"]
b = resultado_1["bias"]
historico_acuracia = resultado_1["historico"]

# Identifica os pontos classificados errados
erros = np.array([
    prever(X[i], w, b) != y[i]
    for i in range(len(X))
])

eixo_x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
eixo_y = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)

xx, yy = np.meshgrid(eixo_x, eixo_y)

# Calcula o valor de z em cada posição da grade
z = w[0] * xx + w[1] * yy + b

plt.figure(figsize=(7, 5))

plt.scatter(
    X[y == 0, 0], X[y == 0, 1],
    label="Classe 0", alpha=0.5, s=15
)
plt.scatter(
    X[y == 1, 0], X[y == 1, 1],
    label="Classe 1", alpha=0.5, s=15
)

# Desenha a linha onde z = 0
plt.contour(xx, yy, z, levels=[0], colors="black")

# Destaca os erros, caso existam
if erros.any():
    plt.scatter(
        X[erros, 0], X[erros, 1],
        facecolors="none", edgecolors="red",
        s=70, label="Classificação incorreta"
    )

plt.xlabel("x₁")
plt.ylabel("x₂")
plt.title("Fronteira aprendida pelo perceptron")
plt.legend()
plt.grid(alpha=0.2)
plt.savefig(FIGURAS / "fig2_fronteira.png", dpi=160, bbox_inches="tight")
plt.close()

epocas = range(1, len(historico_acuracia) + 1)

plt.figure(figsize=(7, 4))
plt.plot(
    epocas,
    np.array(historico_acuracia) * 100,
    marker="o", label="Acurácia no conjunto (classes 0 e 1)"
)

plt.xlabel("Época")
plt.ylabel("Acurácia de treinamento (%)")
plt.title("Evolução do treinamento")
plt.ylim(0, 105)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(FIGURAS / "fig3_acuracia.png", dpi=160, bbox_inches="tight")
plt.close()

print("Épocas executadas:", len(historico_acuracia))
print(f"Acurácia final: {historico_acuracia[-1]:.2%}")
print("Pontos classificados incorretamente:", np.count_nonzero(erros))
print("Pesos finais:", w)
print("Bias final:", b)

