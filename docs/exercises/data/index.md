---
exercise: data
ai_use: "Claude Code (Opus) para reestruturar o repositório ao layout exigido. As análises do relatório são minhas."
---

# Exercise 1: Data

## Exercise 1: Point Clouds — Geometry and Spread in 2D

### A — Generate the clouds

Gerei 400 amostras divididas em 4 classes, 100 em cada. Cada classe vem de uma
gaussiana sorteada de forma independente nos dois eixos, com os parâmetros do
enunciado:

| Classe | Média $\mu$ | Desvio padrão $\sigma$ |
|---|---|---|
| 0 | $[2, 3]$ | $[0{,}8,\ 2{,}5]$ |
| 1 | $[5, 6]$ | $[1{,}2,\ 1{,}9]$ |
| 2 | $[8, 1]$ | $[0{,}9,\ 0{,}9]$ |
| 3 | $[15, 4]$ | $[0{,}5,\ 2{,}0]$ |

Como o $\sigma$ é um vetor e não um único número, cada nuvem fica esticada de
um jeito diferente. A classe 0 tem $\sigma = [0{,}8,\ 2{,}5]$, então é estreita
e alta; já a classe 2 tem $\sigma = [0{,}9,\ 0{,}9]$ e sai quase circular.

A função `generate` recebe um argumento `scale` que multiplica todos os desvios
padrão de uma vez. Aqui ele fica em `1.0`, ou seja, uso os valores do enunciado
sem alteração.

``` { .python .copy title="ex1_clouds.py" linenums="1" }
--8<-- "docs/exercises/data/code/ex1_clouds.py"
```

Rodando o script, as dimensões saem como esperado:

```
X shape: (400, 2), y shape: (400,)
pontos por classe: [100 100 100 100]
```

![Figure 1](figures/fig1_clouds.png)
/// caption
**Figura 1** — as quatro nuvens gaussianas. O X preto marca o centro $\mu$ do
enunciado, e não a média dos pontos que foram sorteados.
///

Na figura dá para ver que a classe 3 fica sozinha em $x_1 \approx 15$, longe das
outras. As classes 0 e 1 se sobrepõem na faixa $x_1 \in [3, 4]$, e a classe 2
encosta na parte de baixo da classe 1.
