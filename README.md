# COMP7607-project

## Abstract

This repository is for HKU COMP7607 Natural Language Processing, Fall 2023 Semester Group Project.

In this study, we expand upon the research conducted by Shah et al. (2023) to find the best-performing model that can classify sentences
from US Federal Open Market Committee (FOMC) historical documents into a hawkish, dovish, or neutral stance. Utilizing the datasets from the original paper, which were extracted from FOMC documents and manually classified, we first replicate the sentence classification outcomes using zero-shot ChatGPT-3.5-Turbo, as well as the fine-tuned Flang-RoBERTa-base, RoBERTa-base and RoBERTalarge models. Our next step involves enhancing the results of the original study by applying prompt engineering to ChatGPT-4-Turbo, exploring new hyperparameters for the RoBERTa-large model, and fine-tuning additional models not covered in the original paper, including XLNet and XLM-RoBERTabase. Our findings reveal that ChatGPT-4-Turbo, when effectively prompted, surpasses the zero-shot performance. Additionally, while XLNet shows superior performance over RoBERTa in analyzing FOMC press conference sentence-split data (PC-S data), RoBERTa remains the top-performing model overall. We further employ the RoBERTa-large, XLNet, and XLM-RoBERTa-base models to calculate the hawkish-tone measure for each historical FOMC document, subsequently devising an investment strategy that yields additional returns over the simple buy-and-hold approach on the QQQ index.

The original paper we base our study on can be found in: 
https://arxiv.org/abs/2305.07972

The original paper's code can be found in:
https://github.com/gtfintechlab/fomc-hawkish-dovish

## Project structure

## Rule-based model

In alignment with the methodology outlined in the original paper, our study employs the rule-based dictionary developed by Gorodnichenko et al. (2021). This dictionary encompasses a variety of terms and keywords related to different monetary policy stances, including topics such as inflation, interest rates, economic activity, and employment. We apply this dictionary specifically to filter the titles of FOMC speeches, intending to identify those speeches that are likely to contain pertinent information on monetary policy. By replicating the rule-based approach from the original paper, we obtain the same results as those presented in the original study.

Our codes and outputs for rule-based model can be found in:


## Zero-shot GPT and prompt engineering

In the original paper, the authors use zero-shot ChatGPT-3.5-Turbo as one of the models to complete the sentence hawkish-dovish classification task.

We first reproduce the result using zero-shot ChatGPT-3.5-Turbo with the original prompt in the paper: 

"Discard all the previous instructions. Behave like you are an expert sentence classifier. Classify the following sentence from FOMC into ‘HAWKISH’, ‘DOVISH’, or ‘NEUTRAL’ class. Label ‘HAWKISH’ if it is corresponding to tightening of the monetary policy, ‘DOVISH’ if it is corresponding to easing of the monetary policy, or ‘NEUTRAL’ if the stance is neutral. Provide the label in the first line and provide a short explanation in the second line."

Then we ask ChatGPT-4-Turbo with the same prompt as in the original paper to observe the zero-shot ChatGPT's performance.

Finally we adopt prompt engineering tips and add our own prompt to ChatGPT-4-Turbo and observe its performance. Our own prompt is:

"Given the FOMC's commitment to maximum employment, stable prices, and moderate long-term interest rates, consider how the sentence reflects these goals. The FOMC aims for transparency in its decisions, acknowledging the fluctuating nature of employment, inflation, and interest rates, and the influence of non-monetary factors. It targets a 2\% inflation rate over the long term, adjusting policy to manage shortfalls in employment and deviations in inflation. The Committee's decisions are informed by a range of indicators, balancing risks and long-term goals, with an annual review of its policy strategy. 
Example: 
Input:  However, other participants noted that the continued subdued trend in wages was evidence of an absence of upward pressure on inflation from the current level of resource utilization. 
Output: DOVISH The sentence suggests that there is no significant wage-induced inflation pressure, which could imply that there is less need for tightening monetary policy. 
Your analysis should incorporate an understanding of the FOMC's principles, particularly how monetary policy actions, including the federal funds rate adjustments, play a role in achieving these objectives and responding to economic disturbances."

Our codes and outputs for zero-shot ChatGPT and prompt engineering can be found in:


## Fine-tuned PLMs

### Reproduction of original paper's experiments on Flang-RoBERTa-base, RoBERTa-base, RoBERTa-large

### Search for additional hyperparameters on RoBERTa-large

### XLNet

### XLM-RoBERTa-base

## Implementing a trading strategy on the sentence classifier
