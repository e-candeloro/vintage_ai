## Sentiment Analysis

Cose già provate o conosciute:
- **VADER** model
- llm hugging face finetunati su social network, + prompt engineering per valutare cosa considerare e cosa no (tipo fake reviews, bot, ecc)
[multilingual sentiment analysis](https://huggingface.co/tabularisai/multilingual-sentiment-analysis)
- [Distilbert-based multi-lingual sentiment classification model](https://huggingface.co/tabularisai/multilingual-sentiment-analysis)

## Topic Modeling

Cose già provate:

- **BERTopic** per topic modelling notebooks:
  - [colab 1](https://colab.research.google.com/drive/1FieRA9fLdkQEGDIMYl0I3MCjSUKVF8C-?usp=sharing)
  - [colab 2](https://colab.research.google.com/drive/1ClTYut039t-LDtlcd-oQAdXWgcsSGTw9?usp=sharing)

- **direttamente LLM** (whisper + summarization dei post, contenuti ecc)

## Value Prediction - Time Series Forecasting
Classic simple models:
- multivariate regression, random forest, trees ecc (no time series) [Regression](https://github.com/gaurav21s/CarPricePrediction/blob/main/Predictive%20Model.ipynb)

- **ARIMA** models
- time series forcasting + RAG
- Foundational Models:
  - [Google TimesFM](https://huggingface.co/google/timesfm-1.0-200m)
  - other SOTA models


## Generative AI + LLM
New AI solutions: 

- RAG models: extract info of classic cars from the capital venture database such as previuos owners, features, prices, show where you can see them from the most important data sources

- [GraphRAG](https://python.langchain.com/docs/tutorials/graph/)
- [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/)
