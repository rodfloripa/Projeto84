
# Geração de Cenários Probabilísticos com Rede Neural Bayesiana e Normalizing Flow

<p align="justify">
Enquanto o **Projeto 78** utilizou **BSTS (Bayesian Structural Time Series)** para explicabilidade de séries temporais e inferência causal, este projeto adota uma abordagem baseada em **Redes Neurais Bayesianas (Bayesian Neural Networks - BNN)** combinadas com **Normalizing Flow** para geração de cenários probabilísticos de demanda.

O objetivo deixa de ser explicar o comportamento passado da série temporal e passa a ser prever possíveis comportamentos futuros considerando a incerteza inerente aos dados. A Rede Neural Bayesiana modela a incerteza dos parâmetros do modelo, enquanto o Normalizing Flow aprende distribuições complexas dos erros de previsão, permitindo representar demandas assimétricas, multimodais e com caudas pesadas.

Essa combinação é especialmente indicada para aplicações de larga escala, como previsão de demanda em milhares de produtos, planejamento de estoques e análise de risco operacional.

</p>

---

# 1. Objetivo

<p align="justify">

Este projeto implementa uma Rede Neural Bayesiana integrada a um modelo de **Normalizing Flow** para gerar distribuições probabilísticas completas da demanda futura. Em vez de produzir apenas um único valor previsto, o modelo estima toda a distribuição de probabilidade das vendas, permitindo construir diferentes cenários e quantificar riscos associados à tomada de decisão.

</p>

---

# 2. Arquitetura do Modelo

<p align="justify">

O modelo é composto por duas partes complementares.

A primeira é uma **Rede Neural Bayesiana (BNN)**, responsável por aprender relações não lineares entre as variáveis explicativas e a demanda.

A segunda é um **Normalizing Flow**, utilizado para modelar distribuições complexas dos resíduos da previsão, permitindo representar comportamentos que não seguem distribuições gaussianas tradicionais.

</p>

```text
Variáveis de Entrada
        │
        ▼
Rede Neural Bayesiana
        │
        ▼
Estimativa da Média
        │
        ▼
Normalizing Flow
        │
        ▼
Distribuição Probabilística
        │
        ▼
Cenários P10 • P50 • P90
```

---

# 3. Comparação entre BSTS e BNN + Normalizing Flow

<p align="justify">

Embora ambos sejam modelos probabilísticos, eles foram desenvolvidos para objetivos diferentes.

</p>

| Característica | BSTS | BNN + Normalizing Flow |
|----------------|------|------------------------|
| Pergunta principal | Qual o efeito de uma campanha? | Qual a distribuição futura da demanda? |
| Tipo de modelo | Modelo estatístico Bayesiano estruturado | Rede Neural Bayesiana |
| Tratamento do tempo | Tendência + sazonalidade + regressão | Necessita atributos temporais e lags |
| Incerteza | Decomposição Bayesiana | Pesos Bayesianos + Flow |
| Escalabilidade | Melhor para poucas séries | Excelente para milhares de séries |
| Dados necessários | Histórico temporal | Grande volume de dados |
| Inferência causal | Sim | Não |
| Distribuições complexas | Limitado | Excelente |
| Overfitting | Baixo | Necessita regularização |
| Aplicação principal | Explicabilidade | Previsão probabilística |

---

# 4. Funcionamento do Modelo

<p align="justify">

O processo inicia com a normalização das variáveis de entrada. Em seguida, essas informações são processadas pela Rede Neural Bayesiana, que estima uma distribuição para os parâmetros do modelo em vez de valores fixos.

Posteriormente, o Normalizing Flow transforma uma distribuição simples em uma distribuição muito mais flexível, capaz de representar comportamentos assimétricos e eventos extremos observados na demanda.

Como resultado, o modelo produz uma distribuição completa das vendas futuras, permitindo calcular percentis, probabilidades de ruptura de estoque e diferentes cenários de risco.

</p>

---

# 5. Principais Trechos do Código

<p align="justify">

Definição dos dados utilizados pela Rede Neural Bayesiana.

</p>

```python
X_data = pm.Data("X_data", X_norm)
```

<p align="justify">

Construção da Rede Neural Bayesiana.

</p>

```python
hidden1 = at.tanh(at.dot(X_data, w1) + b1)
hidden2 = at.tanh(at.dot(hidden1, w2) + b2)
mu = at.flatten(at.dot(hidden2, w3) + b3)
```

<p align="justify">

Aplicação do Normalizing Flow.

</p>

```python
base_dist = pm.Normal.dist(
    mu=mu_original_scale,
    sigma=10
)

obs = pm.NormalizingFlow(
    "obs",
    base_dist=base_dist,
    transforms=[
        pm.distributions.transforms.Exp()
    ],
    observed=y
)
```

<p align="justify">

Geração dos cenários probabilísticos.

</p>

```python
ppc = pm.sample_posterior_predictive(
    trace,
    predictions=True
)
```

---

# 6. Cenários Gerados

<p align="justify">

Após o treinamento do modelo são simulados diferentes cenários de demanda.

Entre eles destacam-se:

- Cenário Base;
- Cenário Otimista;
- Cenário Pessimista.

Para cada cenário são calculados percentis da distribuição prevista, como **P10**, **P50** e **P90**, além da probabilidade de ocorrência de eventos críticos, como demanda inferior ao estoque disponível.

</p>

---

# 7. Quando utilizar BNN + Normalizing Flow

<p align="justify">

Esta abordagem é indicada quando existe grande quantidade de dados históricos, elevado número de produtos ou lojas e necessidade de gerar previsões probabilísticas em larga escala.

Aplicações típicas incluem previsão de demanda, planejamento de estoques, definição de níveis de segurança, análise de risco logístico, simulação de cenários comerciais e otimização de compras.

</p>

---

# 8. Resultado

<p align="justify">

A figura abaixo apresenta um exemplo das distribuições probabilísticas geradas pelo modelo para diferentes cenários de demanda, permitindo visualizar a incerteza da previsão e comparar riscos entre estratégias distintas.

</p>

<br><br><br><br><br>

<p align="center">

**Figura 1 — Distribuições probabilísticas dos cenários gerados (inserir imagem).**

</p>

<br><br><br>

---

# 9. Conclusão

<p align="justify">

A combinação entre **Rede Neural Bayesiana** e **Normalizing Flow** permite produzir previsões probabilísticas muito mais ricas do que modelos tradicionais de regressão, representando adequadamente a incerteza dos parâmetros e distribuições complexas da demanda.

Enquanto o **BSTS**, utilizado no Projeto 78, possui como principal objetivo explicar o comportamento passado da série temporal e medir efeitos causais, a abordagem apresentada neste projeto concentra-se na geração de cenários futuros e na quantificação de riscos em aplicações de grande escala. Dessa forma, os dois modelos são complementares: o BSTS oferece elevada interpretabilidade, enquanto a BNN com Normalizing Flow fornece alta capacidade preditiva para sistemas modernos de planejamento operacional.

</p>
````
