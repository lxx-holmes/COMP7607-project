# COMP7607-project

## Abstract

This repository is for HKU COMP7607 Natural Language Processing, Fall 2023 Semester Final Group Project

In this study, we expand upon the research conducted by Shah et al. (2023) to find the best-performing model that can classify sentences
from US Federal Open Market Committee (FOMC) historical documents into a hawkish, dovish, or neutral stance. Utilizing the datasets from the original paper, which were extracted from FOMC documents and manually classified, we first replicate the sentence classification outcomes using zero-shot ChatGPT-3.5-Turbo, as well as the fine-tuned Flang-RoBERTa-base, RoBERTa-base and RoBERTalarge models. Our next step involves enhancing the results of the original study by applying prompt engineering to ChatGPT-4-Turbo, exploring new hyperparameters for the RoBERTa-large model, and fine-tuning additional models not covered in the original paper, including XLNet and XLM-RoBERTabase. Our findings reveal that ChatGPT-4-Turbo, when effectively prompted, surpasses the zero-shot performance. Additionally, while XLNet shows superior performance over RoBERTa in analyzing FOMC press conference sentence-split data (PC-S data), RoBERTa remains the top-performing model overall. We further employ the RoBERTa-large, XLNet, and XLM-RoBERTa-base models to calculate the hawkish-tone measure for each historical FOMC document, subsequently devising an investment strategy that yields additional returns over the simple buy-and-hold approach on the QQQ index.

The original paper we base our study on can be found in: 
https://arxiv.org/abs/2305.07972

The original paper's code can be found in:
https://github.com/gtfintechlab/fomc-hawkish-dovish

## Project structure

## Reproduction of original paper's experiments

### Rule-based model

### Zero-shot ChatGPT-3.5-Turbo

### Fine-tuned PLMs:Flang-RoBERTa-base, RoBERTa-base, RoBERTa-large

## Model enhancements and experiments on new models

### Zero-shot ChatGPT-4-Turbo

### Prompt engineering on ChatGPT-4-Turbo

### Search for additional hyperparameters on RoBERTa-large model

### Experiment on XLNet model

### Experiment on XLM-RoBERTa-base model

## Implementing a trading strategy on the sentence classifier
