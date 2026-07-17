
# P84 - Geração de Cenários Probabilísticos com Rede Neural Bayesiana e Normalizing Flow

<p align="justify">
Este projeto implementa uma solução para geração de cenários probabilísticos de demanda utilizando uma <b>Rede Neural Bayesiana (Bayesian Neural Network - BNN)</b> combinada com <b>Normalizing Flow</b>. Diferentemente de modelos tradicionais que produzem apenas uma previsão pontual, esta abordagem estima toda a distribuição de probabilidade da demanda futura, permitindo calcular intervalos de confiança, percentis e riscos associados às previsões. Dessa forma, o modelo fornece informações mais completas para apoiar decisões relacionadas a planejamento de estoques, compras, logística e estratégias comerciais.
</p>

---

# 1. Objetivo

<p align="justify">
O objetivo deste projeto é desenvolver um modelo probabilístico capaz de aprender relações não lineares entre variáveis explicativas e demanda, produzindo distribuições completas das previsões em vez de apenas um único valor esperado. A partir dessas distribuições é possível gerar diferentes cenários, calcular probabilidades, estimar riscos e fornecer informações para a tomada de decisão em ambientes sujeitos à incerteza.
</p>

---

# 2. Arquitetura da Solução

<p align="justify">
A solução é composta por duas etapas principais. A primeira utiliza uma Rede Neural Bayesiana para aprender as relações existentes entre as variáveis de entrada e a demanda. A segunda aplica um Normalizing Flow para modelar distribuições complexas dos resíduos da previsão, permitindo representar comportamentos assimétricos, multimodais e com caudas pesadas.
</p>

```text
Variáveis de Entrada
        │
        ▼
Rede Neural Bayesiana
        │
        ▼
Estimativa Inicial
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

# 3. Dados de Entrada

<p align="justify">
O modelo recebe um conjunto de variáveis explicativas capazes de influenciar o comportamento da demanda. Esses atributos são utilizados durante o treinamento para que a Rede Neural Bayesiana aprenda padrões complexos e gere previsões probabilísticas mais precisas.
</p>

| Preço | Gasto com Ads | Temperatura | É Feriado | Vendas |
|------:|--------------:|------------:|:---------:|--------:|
| 99,90 | 1200 | 28 | Sim | 245 |
| 109,90 | 850 | 24 | Não | 187 |
| 89,90 | 1600 | 31 | Sim | 302 |
| 119,90 | 600 | 22 | Não | 161 |
| 95,90 | 1350 | 29 | Sim | 274 |

<p align="justify">
Além das variáveis apresentadas no exemplo, o modelo pode incorporar outras informações relevantes, como dia da semana, promoções, preço da concorrência, sazonalidade, clima, indicadores econômicos, estoque disponível, tráfego do site e quaisquer atributos que contribuam para melhorar a qualidade das previsões.
</p>

---

# 4. Funcionamento do Modelo

<p align="justify">
Inicialmente os dados são normalizados para facilitar o treinamento da Rede Neural Bayesiana. Em seguida, a rede estima distribuições para seus pesos em vez de valores fixos, representando a incerteza do modelo. Posteriormente, o Normalizing Flow transforma uma distribuição probabilística simples em uma distribuição muito mais flexível, capaz de representar diferentes formatos observados na demanda real. Ao final do processo, o modelo gera uma distribuição completa das vendas futuras, possibilitando calcular percentis, intervalos de confiança e probabilidades de ocorrência de diferentes cenários.
</p>

---

# 5. Principais Trechos do Código

<p align="justify">
Definição dos dados utilizados pelo modelo.
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
Aplicação do Normalizing Flow para modelagem da distribuição da demanda.
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
Geração das previsões probabilísticas.
</p>

```python
ppc = pm.sample_posterior_predictive(
    trace,
    predictions=True
)
```

---

# 6. Configuração dos Cenários

<p align="justify">
Após o treinamento da Rede Neural Bayesiana, são criados diferentes cenários alterando apenas as variáveis de entrada do modelo. Como a rede aprendeu a relação entre essas variáveis e a demanda, cada alteração produz uma nova distribuição probabilística das vendas. Dessa forma, é possível avaliar como diferentes condições operacionais impactam a previsão da demanda.
</p>

| Cenário | Preço | Gasto com Ads | Temperatura | Feriado | Objetivo |
|---------|:-----:|:-------------:|:-----------:|:--------:|----------|
| Base | Valor médio | Valor médio | Valor médio | Valor médio | Representar a situação normal de operação. |
| Otimista | Mantido | Aumentado | Maior | Sim | Simular condições favoráveis para aumento das vendas. |
| Pessimista | Aumentado | Reduzido | Mantida | Não | Simular condições desfavoráveis para redução da demanda. |

<p align="justify">
No cenário base, todas as variáveis permanecem próximas de seus valores médios observados durante o treinamento. Esse cenário representa o comportamento esperado da demanda caso nenhuma alteração significativa ocorra nas condições de operação.
</p>

<p align="justify">
No cenário otimista, o investimento em publicidade é aumentado, a temperatura é elevada e considera-se a ocorrência de um feriado. Essas modificações tendem a aumentar a demanda prevista pelo modelo, deslocando a distribuição de probabilidades para valores maiores de vendas.
</p>

<p align="justify">
No cenário pessimista, o preço do produto é elevado e o investimento em publicidade é reduzido. Essas alterações representam condições menos favoráveis ao consumo, fazendo com que a distribuição prevista das vendas seja deslocada para valores menores e aumentando a probabilidade de ocorrência de baixa demanda.
</p>

---

# 7. Aplicações

<p align="justify">
A abordagem pode ser aplicada em problemas de previsão de demanda, planejamento de estoques, definição de níveis de segurança, compras, logística, planejamento comercial, análise de risco, simulação de cenários e suporte à tomada de decisão. Sua capacidade de representar distribuições probabilísticas completas permite avaliar diferentes possibilidades futuras em vez de depender exclusivamente de uma previsão pontual.
</p>

---

# 8. Resultado

<p align="justify">
A figura abaixo apresenta um exemplo das distribuições probabilísticas geradas pelo modelo para diferentes cenários de demanda. A visualização permite comparar a dispersão das previsões, os intervalos de confiança e os níveis de risco associados a cada cenário analisado.
</p>
<p align="center">
  <img src="https://github.com/rodfloripa/Projeto84/blob/main/fig.png">
</p>
<br>
<p align="center">
<b>Figura 1 — Distribuições probabilísticas dos cenários gerados.</b>
</p>

| Cenário | P10 | P50 | P90 |
|---------|-----:|-----:|-----:|
| Base | 146 | 196 | 243 |
| Otimista | 149 | 199 | 247 |
| Pessimista | 70 | 119 | 167 |

<p align="justify">
O percentil <b>P10</b> representa um cenário conservador, indicando que existe aproximadamente <b>10% de probabilidade de a demanda ser inferior ao valor apresentado</b>. O <b>P50</b> corresponde à mediana da distribuição, ou seja, existe 50% de probabilidade de a demanda ficar abaixo desse valor e 50% acima. Já o <b>P90</b> representa um cenário otimista, indicando que aproximadamente 90% das previsões ficam abaixo desse valor e apenas 10% o ultrapassam.
</p>

<br><br><br>

---

# 9. Conclusão

<p align="justify">
A combinação entre Redes Neurais Bayesianas e Normalizing Flow permite construir modelos capazes de representar simultaneamente relações não lineares entre as variáveis de entrada, incertezas dos parâmetros e distribuições complexas da demanda. Em vez de produzir apenas um único valor previsto, a abordagem fornece uma distribuição probabilística completa, permitindo calcular intervalos de confiança, percentis e probabilidades de diferentes cenários. Essas características tornam o modelo uma ferramenta importante para aplicações que exigem previsão sob incerteza e suporte à tomada de decisão baseada em risco.
</p>
