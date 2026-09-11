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

### B — More or less spread out

Repeti a geração quatro vezes, multiplicando todos os desvios padrão por
$s \in \{0{,}5;\ 1{,}0;\ 2{,}0;\ 4{,}0\}$. As médias não mudam, só o
espalhamento.

![Figure 2](figures/fig2_scales.png)
/// caption
**Figura 2** — as mesmas quatro classes com os desvios multiplicados por $s$.
Os quatro painéis usam os mesmos limites de eixo, calculados sobre todas as
escalas, senão a comparação visual não valeria nada.
///

#### Razão de separação

Para $s = 1$, calculei $r_{ij} = \dfrac{\lVert \mu_i - \mu_j \rVert}{\bar\sigma_i + \bar\sigma_j}$
nos seis pares, usando $\bar\sigma$ como a média dos dois desvios da classe:

| Par | $r_{ij}$ |
|---|---|
| 0 e 1 | **1,326** |
| 1 e 2 | 2,380 |
| 0 e 2 | 2,480 |
| 2 e 3 | 3,542 |
| 1 e 3 | 3,642 |
| 0 e 3 | 4,496 |

O menor é o par **0 e 1**, com $r_{01} = 1{,}326$ — o que bate com o que se vê
na Figura 1, onde são as duas nuvens que se tocam.

Para $s = 2$ a previsão é $0{,}663$, ou seja, metade. O raciocínio é que
escalar $s$ multiplica os desvios mas não mexe nas médias: o numerador de
$r_{ij}$ fica igual e o denominador dobra. Rodando com $s = 2$ o valor dá
exatamente $0{,}663$, confirmando.

#### Taxa de mistura

Calculei a fração de pontos cujo centro verdadeiro mais próximo pertence a outra
classe:

| $s$ | Taxa de mistura |
|---|---|
| 0,5 | 0,000 |
| 1,0 | 0,068 |
| 2,0 | 0,225 |
| 4,0 | 0,417 |

![Figure 3](figures/fig3_mixing_rate.png)
/// caption
**Figura 3** — taxa de mistura em função da escala aplicada aos desvios.
///

Com $s = 0{,}5$ nenhum ponto cai do lado errado: as nuvens ficam tão apertadas
que se separam completamente. A partir daí a mistura cresce rápido, e em
$s = 4$ quase metade dos pontos já está mais perto do centro de outra classe.

Para achar o limiar, usei o $r_{ij}$ do par mais apertado, que é o 0-1. Quando
$r_{01} = 1$, a distância entre os dois centros fica igual à soma dos
espalhamentos médios, ou seja, acabou a faixa livre entre as nuvens. Como
$r_{01}(s) = 1{,}326 / s$, isso dá $s \approx 1{,}33$. Então é entre $s = 1$ e
$s = 2$ que o par 0-1 deixa de ser linearmente separável, o que bate com o salto
da taxa de mistura de 0,068 para 0,225 nesse intervalo.

### C — Analysis

**Como as classes se sobrepõem.** Na Figura 1, a classe 3 fica isolada em
$x_1 \approx 15$ e dá para separar dela com uma reta vertical. O problema são as
classes 0 e 1, que se encostam em $x_1 \in [3, 4]$ — e o $r_{01} = 1{,}326$
confirma que é o par mais apertado dos seis. A classe 2 encosta na parte de
baixo da classe 1, mas com bem mais folga.

Uma reta só não separa as quatro classes. Consigo traçar uma que isola a classe
3, mas aí as outras três ficam todas do mesmo lado, então precisaria de pelo
menos mais uma. E como as nuvens têm formatos bem diferentes (a classe 0 é alta
e estreita, a 2 é quase circular), acho que a fronteira ideal entre elas nem
seria reta.

**Esboço das fronteiras.** Desenhei as fronteiras usando o critério de centro
mais próximo, que foi a divisão mais simples que consegui pensar respeitando os
quatro centros:

![Fronteiras de decisão](figures/fig1_boundaries.png)
/// caption
**Figura 1b** — as mesmas nuvens, com as regiões de decisão do classificador de
centro mais próximo. As linhas pretas são as fronteiras.
///

Dá para ver os pontos que caem do lado errado, quase todos na divisa entre o
azul e o laranja. São exatamente os 6,8% que contei na taxa de mistura para
$s = 1$.

**Ligação com o item B.** Aumentar o espalhamento não muda as fronteiras, já que
elas dependem só das médias. O que muda é quanto de cada nuvem passa para o lado
errado. Em $s = 0{,}5$ as nuvens ficam compactas e nenhum ponto cruza; quando $s$
cresce, as caudas começam a invadir a região vizinha.

Esses pontos que invadem são erro que não tem como evitar: eles caem numa região
onde outra classe é mais provável, então nenhum classificador acerta todos sem
decorar o treino. É isso que a taxa de mistura está medindo, e por isso ela
depende da razão entre distância e espalhamento, e não só da distância.
