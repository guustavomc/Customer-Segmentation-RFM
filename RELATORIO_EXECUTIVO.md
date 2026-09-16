# Relatório Executivo — Segmentação de Clientes RFM

**Base:** UCI Online Retail Dataset (e-commerce, Reino Unido, dez/2010–dez/2011) · 4.251 clientes analisados
**Algoritmo final:** K-Means (k=5) · **Fontes:** [`outputs/perfil_clusters.csv`](outputs/perfil_clusters.csv), [`outputs/comparacao_algoritmos.csv`](outputs/comparacao_algoritmos.csv), [`segmentacao_clientes_rfm.ipynb`](segmentacao_clientes_rfm.ipynb), [`outputs/dashboard.html`](outputs/dashboard.html)

---

## 1. Sumário Executivo

A análise RFM (Recency, Frequency, Monetary) segmentou a base de 4.251 clientes em **5 grupos comportamentais** usando K-Means. Principais descobertas:

- **Champions** são apenas 13,9% da base (589 clientes) mas geram **50,6% da receita** (£3,05M) — concentração de valor clássica de Pareto.
- **At Risk** é o segundo maior bloco de receita (23,8%, £1,44M) e o segundo maior em clientes (24,4%), mas já está com recência alta (~100 dias sem comprar) — receita real em risco de churn no curto prazo.
- **Lost** e **New Customers** somam 44,5% dos clientes mas apenas 9,5% da receita — baixo valor atual, mas por razões opostas (um já saiu, o outro ainda não maturou).
- **Loyal Customers** (17,2% dos clientes, 16,1% da receita) é a base mais estável e recente do negócio, com bom potencial de upsell.

**Top 3 recomendações prioritárias:**
1. Campanha de reativação com prazo curto para **At Risk** — maior valor monetário em risco imediato de churn.
2. Programa VIP/fidelidade para **Champions** — proteger a metade da receita concentrada em 14% da base.
3. Fluxo de onboarding/nutrição para **New Customers** — converter antes que migrem para "Lost".

---

## 2. Metodologia

**Pipeline:** limpeza de transações inválidas (sem `CustomerID`, cancelamentos, quantidades/preços não positivos) → cálculo de RFM por cliente → remoção de outliers extremos de Monetary (percentis 1%–99%) → `log1p` nas três variáveis (Recency/Frequency/Monetary têm assimetria de 1,3 a 19,3 — sem essa transformação, os clusters colapsam artificialmente em torno da cauda longa de poucos clientes muito atípicos) → padronização com `RobustScaler`.

**Escolha do algoritmo final:** os três algoritmos exigidos foram implementados e comparados (silhouette, Davies-Bouldin, Calinski-Harabasz, tempo de execução — ver `comparacao_algoritmos.csv`). K-Means (k=5) foi escolhido por entregar qualidade equivalente ao Hierárquico (silhouette 0,317 vs. 0,270) a um custo computacional ~11x menor, e por atribuir todos os clientes a um segmento — diferente do DBSCAN, que encontrou apenas 3 clusters de qualidade inferior (silhouette 0,169) e classificou 135 clientes (3,2%) como ruído, inutilizável para uma ação de marketing dirigida a 100% da base.

**Escolha de k:** o Silhouette Score isolado indicava k=2 (separação rasa "alto valor vs. resto"), enquanto o Elbow Method (cotovelo da curva de inércia) indicava k=5. Adotamos k=5 por preservar a granularidade de negócio exigida pelo framework RFM clássico (Champions/Loyal/At Risk/New/Lost) — essa divergência entre os dois métodos é registrada explicitamente no pipeline como uma decisão metodológica deliberada, não uma omissão.

**Validação:** estabilidade do K-Means testada com 10 seeds diferentes (ARI médio = 0,966, indicando clusters praticamente idênticos independente da inicialização); convergência em 17 iterações em média, bem abaixo do limite de 300. Redução dimensional via PCA preserva 94,1% da variância em 2D, validando as visualizações do Item 2.

**Limitações identificadas:** (1) a base cobre apenas 12 meses de um único mercado (Reino Unido), sem sazonalidade de anos anteriores nem dados de categoria de produto/canal — a segmentação reflete só comportamento transacional; (2) RFM não distingue clientes atacadistas de varejo, um viés conhecido deste dataset; (3) a escolha de k por Elbow é uma decisão de trade-off entre rigor estatístico (Silhouette) e utilidade de negócio — deve ser revalidada com a área de marketing periodicamente, já que RFM é sensível à data de referência e se degrada com o tempo.

---

## 3. Resultados e Insights

| Segmento | Clientes | % Base | Receita | % Receita | Recência média | Frequência média | Ticket médio anual |
|---|---|---|---|---|---|---|---|
| **Champions** | 589 | 13,9% | £3,05M | 50,6% | 18 dias | 12,7 pedidos | £5.184 |
| **At Risk** | 1.039 | 24,4% | £1,44M | 23,8% | 100 dias | 3,3 pedidos | £1.383 |
| **Loyal Customers** | 730 | 17,2% | £972K | 16,1% | 13 dias | 4,3 pedidos | £1.332 |
| **Lost** | 1.025 | 24,1% | £289K | 4,8% | 230 dias | 1,3 pedidos | £282 |
| **New Customers** | 868 | 20,4% | £285K | 4,7% | 38 dias | 1,5 pedidos | £328 |

**Champions** compram com frequência quase mensal (12,7 pedidos/ano) e têm ticket médio anual 3,7x maior que Loyal Customers — o segmento mais rentável por cliente, de longe. Concentrar apenas 14% da base em 50% da receita é tanto uma força (alta lucratividade por cliente) quanto um risco de concentração: a perda de uma fração desse grupo tem impacto desproporcional.

**At Risk** é o achado mais acionável: esses clientes já compraram bem (ticket médio £1.383, mais alto que Loyal Customers) e com frequência razoável (3,3 pedidos), mas não compram há ~100 dias — o dobro da recência de Loyal Customers. Isso não é um cliente de baixo valor, é um cliente de valor comprovado se afastando. Recuperar mesmo 20% desse grupo preserva algo entre £250K–£300K de receita anual.

**Loyal Customers** têm o perfil mais saudável do ponto de vista de recência (13 dias, o menor de todos os grupos) mas frequência ainda moderada (4,3 pedidos/ano) e ticket relativamente baixo — é a base com maior potencial de crescimento via cross-sell/upsell, pois já está engajada e recente.

**New Customers** têm recência razoável (38 dias) mas frequência muito baixa (1,5 pedidos) — coerente com clientes early-lifecycle que ainda não repetiram compra. A pergunta de negócio é se este grupo vai migrar para Loyal Customers ou para Lost, e isso é decidido nos primeiros 60–90 dias.

**Lost** é o grupo de menor prioridade: 230 dias de recência média e apenas 1,3 pedido/ano indicam clientes que provavelmente não vão reativar organicamente. Representam 24% da base mas menos de 5% da receita — o oposto de Champions.

**Oportunidade de negócio principal:** a receita da empresa está polarizada entre "muito engajado" (Champions+Loyal = 31% da base, 66,7% da receita) e "pouco engajado" (At Risk+Lost+New = 69% da base, 33,3% da receita). Investir em mover clientes de At Risk e New Customers para Loyal tem mais retorno esperado do que tentar recuperar Lost.

---

## 4. Recomendações Estratégicas

**1. Win-back para At Risk (prioridade máxima).** Cupom de reativação com prazo curto (7–14 dias) e comunicação personalizada por categoria de compra anterior. Meta sugerida: reativar 15–20% do segmento em 90 dias, o que preserva ~£220K–£290K de receita anual que hoje está em risco de virar "Lost". Métrica de acompanhamento: % de At Risk que volta a comprar dentro de 90 dias da campanha.

**2. Programa VIP para Champions.** Acesso antecipado a lançamentos, atendimento dedicado e benefícios de frete/fidelidade. Como este grupo já gera a maior parte da receita, o objetivo não é crescer o segmento, e sim **não perder nenhum cliente dele** — o custo de retenção aqui é muito menor que o custo de substituição (aquisição de um novo cliente de alto valor). Métrica: taxa de churn de Champions trimestre a trimestre (meta: manter abaixo de 5%).

**3. Onboarding e incentivo à segunda compra para New Customers.** E-mail/SMS educativo nos primeiros 30 dias, cupom para a segunda compra e recomendação de produtos complementares ao primeiro pedido. Meta: elevar a taxa de segunda compra em 90 dias — hoje a frequência média de 1,5 pedido sugere que a maioria não repete. Métrica: % de New Customers que faz 2ª compra em 90 dias.

**4. Cross-sell/upsell para Loyal Customers.** Programa de fidelidade com descontos progressivos por volume e recomendações baseadas em histórico — este é o segmento com melhor relação custo/benefício de investimento, pois já está engajado e recente.

**5. Baixo investimento em Lost.** Uma única campanha de win-back de baixo custo (e-mail, sem desconto agressivo) é suficiente; não vale alocar orçamento significativo de CRM neste grupo — o retorno esperado é baixo dado o padrão de 230 dias de inatividade.

**Viabilidade:** todas as recomendações usam canais e mecanismos já comuns em CRM de e-commerce (e-mail marketing, cupons, programas de fidelidade), sem necessidade de investimento em nova infraestrutura. O maior risco de execução é operacional (capacidade da equipe de marketing/CRM de rodar 3 campanhas segmentadas simultaneamente), não técnico.

**Próximo passo analítico:** revalidar esta segmentação trimestralmente (RFM é sensível à data de referência) e, se houver dados adicionais (categoria de produto, canal de aquisição, geografia), testar uma segunda rodada de clustering com essas features para refinar as recomendações por subsegmento.
