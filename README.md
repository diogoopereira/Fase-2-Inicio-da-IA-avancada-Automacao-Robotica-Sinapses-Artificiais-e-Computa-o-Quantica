# CardioIA – Fase 2: Diagnóstico Automatizado

## Projeto Acadêmico – FIAP | Inteligência Artificial

Na **Fase 2** do CardioIA simulamos o "estetoscópio digital": um módulo que lê relatos de pacientes, reconhece sintomas, sugere diagnósticos e classifica o risco de cada caso. Tudo é feito com ferramentas simples e dados bem organizados, e em cada etapa analisamos os vieses e as limitações das soluções.

| Parte | O que faz | Técnica |
|---|---|---|
| [Parte 1](parte1/) | Extrai sintomas de relatos e sugere um diagnóstico | Mapa de conhecimento (ontologia) + busca de expressões |
| [Parte 2](parte2/) | Classifica frases em **alto** ou **baixo risco** | TF-IDF + Regressão Logística |
| [Ir Além 2](ir-alem-2/) | Classifica imagens de ECG em **normal** ou **anormal** | Rede neural MLP (Keras) |

> Projeto acadêmico com dados simulados ou públicos e anonimizados. Nenhuma das soluções substitui avaliação médica.

---

## Vídeo de demonstração

- **Fase 2 (Partes 1 e 2):** [assista no YouTube](https://youtu.be/lD8X5Hb3fQE)
- **Ir Além 2:** link no [README do Ir Além 2](ir-alem-2/README.md)

---

## Integrantes

- Filipe Augusto Lima Silva
- Laísa Cristina Capodifoglio Andrade
- Johnathan da Cruz Gatti
- Diogo Ferreira Pereira
- André Victor Gonçalves Toledo

## Professores

### Tutor

- Leonardo Ruiz Orabona

### Coordenador

- André Godoi Chiovato

---

## Estrutura do repositório

```text
CardioIA-Fase2/
├── README.md
├── requirements.txt
├── parte1/
│   ├── frases_sintomas.txt          # 10 relatos de pacientes
│   ├── mapa_conhecimento.csv        # sintoma_1, sintoma_2 → doença (34 associações, 10 doenças)
│   └── extrator_diagnostico.ipynb   # leitura, extração de sintomas e diagnóstico
├── parte2/
│   ├── frases_risco.csv             # 100 frases rotuladas (50 alto risco, 50 baixo risco)
│   └── classificador_risco.ipynb    # TF-IDF, treino, avaliação e análise de vieses
└── ir-alem-2/
    ├── README.md
    ├── classificador_ecg_mlp.ipynb  # geração de imagens, pré-processamento e MLP
    └── exemplos/                    # amostras das imagens de ECG usadas
```

---

## Como executar

Os notebooks já estão salvos com as saídas, então dá para ver todos os resultados direto no GitHub. Para rodar de novo:

```bash
git clone <url-deste-repositorio>
cd CardioIA-Fase2
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Use **Python 3.10 a 3.13**: o TensorFlow, usado no Ir Além 2, ainda não suporta o 3.14. Cada notebook lê os arquivos da própria pasta. No **Google Colab**, envie para o ambiente o `.ipynb` junto com os arquivos `.txt` e `.csv` da mesma pasta. O notebook do Ir Além 2 baixa os dados sozinho.

---

## Parte 1 – Extração de sintomas e sugestão de diagnóstico

**Arquivos:** [`frases_sintomas.txt`](parte1/frases_sintomas.txt), [`mapa_conhecimento.csv`](parte1/mapa_conhecimento.csv) e [`extrator_diagnostico.ipynb`](parte1/extrator_diagnostico.ipynb).

Os **10 relatos** simulam pacientes diferentes. Cada um diz **o que sente**, **quando começou** e **como isso afeta a rotina**. Exemplo:

> "Há três semanas sinto um aperto no peito sempre que subo escadas, que melhora com repouso, e agora evito ir a pé até o mercado."

O **mapa de conhecimento** segue a estrutura `sintoma_1, sintoma_2, doenca_associada`, com 34 associações para 10 doenças cardiovasculares. As expressões imitam o jeito de falar do paciente ("coração disparado", "vários travesseiros", "quase caí").

**Como o código funciona:**

1. Frases e sintomas vão para minúsculas e perdem os acentos ("tórax" = "torax").
2. Cada expressão do mapa é procurada dentro da frase.
3. Cada doença ganha 1 ponto por sintoma encontrado. A mais pontuada é o **diagnóstico sugerido**, e as demais aparecem como **hipóteses secundárias**, como em um diagnóstico diferencial.

**Resultado: 10/10 relatos com o diagnóstico esperado.**

| # | Principais sintomas identificados | Diagnóstico sugerido |
|---|---|---|
| 1 | dor forte no peito, irradia para o braço, suor frio, náusea | Infarto agudo do miocárdio |
| 2 | aperto no peito, subo escadas, melhora com repouso | Angina estável |
| 3 | pernas inchadas, falta de ar ao deitar, vários travesseiros | Insuficiência cardíaca |
| 4 | coração disparado, batimentos irregulares | Fibrilação atrial |
| 5 | batimentos lentos, tontura, quase caí, fraqueza | Bradicardia |
| 6 | dor de cabeça na nuca, visão embaçada, pressão alta | Hipertensão arterial |
| 7 | piora ao respirar fundo, inclino para frente, depois de uma gripe | Pericardite |
| 8 | virose, febre, cansaço extremo, palpitações | Miocardite |
| 9 | desmaiei, durante o treino, morte súbita, falta de ar ao correr | Cardiomiopatia hipertrófica |
| 10 | falta de ar súbita, dor ao respirar, viagem de avião | Embolia pulmonar |

**Limitações testadas no notebook:**

- **Negação:** "*não* sinto dor no peito" ainda casa com "dor no peito".
- **Termos técnicos** ("precordialgia", "dispneia") ficam fora do mapa.
- **Variações de escrita** ("dor no tórax que aperta" ≠ "aperto no tórax") não são reconhecidas.

---

## Parte 2 – Classificador de risco

**Arquivos:** [`frases_risco.csv`](parte2/frases_risco.csv) e [`classificador_risco.ipynb`](parte2/classificador_risco.ipynb).

- **Base:** 100 frases escritas pelo grupo (`frase,situacao`), 50 de alto risco e 50 de baixo risco. De propósito, algumas frases de baixo risco mencionam "peito" ou "coração acelerado" (dor muscular depois do supino, café demais), para o modelo não aprender só atalhos.
- **Vetorização:** TF-IDF com palavras e pares de palavras (`ngram_range=(1, 2)`) e sem acentos. As stopwords **não** foram removidas, porque listas comuns em português (como a do NLTK) apagam o "não".
- **Modelo:** Regressão Logística (scikit-learn), escolhida por ser simples e interpretável.
- **Avaliação:** 75% das frases para treino e 25% para teste, com estratificação, mais validação cruzada de 5 dobras.

| Métrica | Resultado |
|---|---|
| Acurácia no teste (25 frases) | **84%** |
| Validação cruzada (5 dobras) | **86% ± 7%** |
| Recall de alto risco | 77% (3 falsos negativos em 13) |

**Padrões e distorções encontrados:**

- O termo mais forte para alto risco é **"peito"**. Por isso "*não* tenho dor no peito" e "meu *avô* está com dor no peito" viram alto risco.
- **"desmaio"** não existia no treino (só "desmaiei"), e a frase com desmaio foi classificada como baixo risco.
- O **"ar"** de "ar-condicionado" foi lido como o "ar" de "falta de ar".
- Os termos que mais indicam baixo risco são "quando", "leve", "mas" e "depois". Isso é **viés de escrita** de quem montou a base, não sintoma.
- Um **infarto com sintomas atípicos** (costas, cansaço, enjoo), apresentação mais comum em mulheres, idosos e diabéticos, foi classificado como baixo risco. É um viés com impacto real sobre esses grupos.

Na triagem, o erro mais grave é o **falso negativo**: mandar um paciente grave para o fim da fila. Em um sistema real, priorizaríamos o recall de alto risco em vez da acurácia. A análise completa está no final do notebook.

---

## Ir Além 2 – Diagnóstico visual de ECG com rede neural MLP

Reaproveitamos o pipeline de imagens de ECG da Fase 1 (PTB-XL) e fizemos duas correções: tiramos o vazamento de rótulo (o nome da classe estava escrito no título da imagem) e fixamos a escala vertical, para preservar a amplitude do sinal. Com 1.200 imagens em tons de cinza de 128 × 128, uma MLP em Keras (256 → 64 → 1) chegou a **69,7% de acurácia** no teste, separando pacientes com os folds oficiais do PTB-XL. Detalhes, análise e vídeo estão no [README do Ir Além 2](ir-alem-2/README.md).

---

## Relação com a Fase 1

A Fase 2 reaproveita a [base da Fase 1](https://github.com/andrevgtoledo/CardioIA-Fase1):

- As doenças do mapa de conhecimento conversam com as classes do PTB-XL usadas na Fase 1: **MI** → infarto; **CD** → fibrilação atrial e bradicardia; **HYP** → cardiomiopatia hipertrófica. As anginas vêm da variável `dor_peito` da planilha `cardioia_dados_numericos_iot.xlsx`.
- Os fatores de risco da planilha (tabagismo, diabetes, hipertensão, histórico familiar) aparecem nas frases de alto risco da Parte 2.
- O notebook `gerar_imagens_ecg.ipynb` da Fase 1 é a base do Ir Além 2.

---

## Governança, ética e limitações

- **Dados:** as frases das Partes 1 e 2 foram criadas pelo grupo e não pertencem a pacientes reais. O PTB-XL é público e anonimizado. Nenhum dado pessoal identificável é usado, em linha com a LGPD.
- **Viés de quem rotula:** o mesmo grupo escreveu e rotulou as frases, e o modelo aprendeu o estilo de escrita do grupo. Uma base real precisa de relatos de pessoas diversas e de rotulagem por profissionais de saúde.
- **Representatividade:** a base tem poucos exemplos de apresentações atípicas, o que prejudica justamente os grupos que mais as apresentam.
- **Uso responsável:** as três soluções são ferramentas de **apoio**. A decisão clínica final é sempre de um profissional de saúde.

---

## Tecnologias

Python, pandas, scikit-learn, Keras/TensorFlow, WFDB, Pillow, Matplotlib, Jupyter e GitHub.

## Referências

- Wagner, P. et al. *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data, 2020. https://physionet.org/content/ptb-xl/1.0.3/
- scikit-learn: https://scikit-learn.org
- Keras: https://keras.io

---

**FIAP – Inteligência Artificial | CardioIA – Fase 2: Diagnóstico Automatizado**
