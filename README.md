
#  Geração de Cenários Probabilísticos com Rede Neural Bayesiana e Normalizing Flow

<p align="justify">
Enquanto o <b>Projeto 78</b> utilizou <b>BSTS (Bayesian Structural Time Series)</b> para explicabilidade de séries temporais e inferência causal, este projeto adota uma abordagem baseada em <b>Redes Neurais Bayesianas (Bayesian Neural Networks - BNN)</b> combinadas com <b>Normalizing Flow</b> para geração de cenários probabilísticos de demanda. O objetivo deixa de ser explicar o comportamento passado da série temporal e passa a ser prever possíveis comportamentos futuros considerando a incerteza inerente aos dados. A Rede Neural Bayesiana modela a incerteza dos parâmetros do modelo, enquanto o Normalizing Flow aprende distribuições complexas dos erros da previsão, permitindo representar demandas assimétricas, multimodais e com caudas pesadas.
</p>

---

# 1. Objetivo

<p align="justify">
Este projeto implementa uma Rede Neural Bayesiana integrada a um modelo de <b>Normalizing Flow</b> para gerar distribuições probabilísticas completas da demanda futura. Em vez de produzir apenas um único valor previsto, o modelo estima toda a distribuição de probabilidade das vendas, permitindo construir diferentes cenários, calcular percentis, estimar riscos operacionais e apoiar decisões relacionadas a estoque, compras e planejamento comercial.
</p>

---

# 2. Arquitetura da Solução

<p align="justify">
A arquitetura é composta por uma Rede Neural Bayesiana responsável por aprender relações não lineares entre as variáveis de entrada e a demanda. Em seguida, um Normalizing Flow transforma uma distribuição simples em uma distribuição probabilística muito mais flexível, permitindo representar comportamentos que dificilmente seriam modelados por distribuições gaussianas tradicionais.
</p>

```text
Variáveis de Entrada
        │
        ▼
Rede Neural Bayesiana
        │
        ▼
Estimativa da Demanda
        │
        ▼
Normalizing Flow
        │
        ▼
Distribuição Probabilística
        │
        ▼
P10 • P50 • P90
```

---

# 3. BSTS vs BNN + Normalizing Flow

<p align="justify">
Embora ambos sejam modelos probabilísticos, eles possuem objetivos bastante diferentes. O BSTS foi desenvolvido para explicar séries temporais e medir efeitos causais, enquanto a BNN com Normalizing Flow foi desenvolvida para gerar previsões probabilísticas em larga escala.
</p>

| Característica | BSTS | BNN + Normalizing Flow |
|----------------|------|------------------------|
| Pergunta principal | Qual o efeito da campanha? | Qual a distribuição da demanda? |
| Tipo de modelo | Modelo estatístico estruturado | Rede Neural Bayesiana |
| Tratamento temporal | Tendência + sazonalidade + regressão | Necessita atributos temporais |
| Incerteza | Inferência Bayesiana | Pesos Bayesianos + Flow |
| Escalabilidade | Poucas séries | Milhares de séries |
| Inferência causal | Sim | Não |
| Distribuições complexas | Limitado | Excelente |
| Aplicação | Explicabilidade | Previsão probabilística |

---

# 4. Funcionamento do Modelo

<p align="justify">
Inicialmente os dados são normalizados para facilitar o treinamento da Rede Neural Bayesiana. A rede estima uma distribuição para seus pesos em vez de valores determinísticos, permitindo representar a incerteza do modelo. Posteriormente, o Normalizing Flow aprende uma transformação probabilística capaz de representar distribuições complexas dos resíduos da previsão. Como resultado, o modelo produz uma distribuição completa da demanda futura em vez de apenas uma previsão pontual.
</p>

---

# 5. Principais Trechos do Código

<p align="justify">
Definição dos dados de entrada.
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
Após o treinamento do modelo são simulados diferentes cenários de demanda. Neste projeto foram utilizados um cenário base, um cenário otimista e um cenário pessimista. Para cada situação são calculados percentis como P10, P50 e P90, permitindo estimar riscos, intervalos prováveis de vendas e probabilidades de ruptura ou excesso de estoque.
</p>

---

# 7. Quando Utilizar BNN + Normalizing Flow

<p align="justify">
Esta abordagem é indicada quando existe grande volume de dados históricos, milhares de produtos, múltiplas lojas ou diversas variáveis explicativas. Também é adequada quando o objetivo é produzir previsões probabilísticas para apoiar decisões de compras, planejamento de estoque, logística, simulação de cenários e análise de risco operacional.
</p>

---

# 8. Resultado

<p align="justify">
A figura abaixo apresenta um exemplo das distribuições probabilísticas geradas para os diferentes cenários simulados, permitindo comparar a dispersão da demanda prevista, os intervalos de confiança e os níveis de risco associados a cada estratégia avaliada.
</p>

<br><br><br><br><br><br>

<p align="center">
<b>Figura 1 — Distribuições probabilísticas dos cenários gerados (inserir imagem).</b>
</p>

<br><br><br>

---

# 9. Conclusão

<p align="justify">
A combinação entre Redes Neurais Bayesianas e Normalizing Flow permite construir modelos capazes de representar simultaneamente relações não lineares, incertezas dos parâmetros e distribuições complexas da demanda. Diferentemente do BSTS, cujo foco é explicar o comportamento da série temporal e medir efeitos causais, a abordagem apresentada neste projeto concentra-se na geração de cenários futuros e na quantificação probabilística dos riscos. Dessa forma, o modelo torna-se uma ferramenta valiosa para planejamento operacional, previsão de demanda em larga escala e suporte à tomada de decisão baseada em incerteza.
</p>
````
