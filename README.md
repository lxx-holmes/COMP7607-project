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

[Rule-based](Rule-based/)


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

[GPT](GPT/)

## Fine-tuned PLMs

### Reproduction of original paper's experiments on Flang-RoBERTa-base, RoBERTa-base, RoBERTa-large

We choose three fine-tuned PLMs from the original paper to reproduce the experiment result: the Flang-RoBERTa-base model trained on MM-S data, which is the best-performing finance domain-specific language model; the RoBERTa-base trained on Combined-S data, which is the best-performing base size model; and the RoBERTa-large model trained on combined data, which is the overall best-performing model. We reproduce the results and compare with the authors' original results. 

In the original paper, Flang-RoBERTa-base trained on MM-S data is the best-performing finance domain-specific model, achieving test F1 of 0.6854. We use the same set of hyperparameters on the MM-S data, and get a test F1 of 0.6692.

The codes for reproduction of Flang-RoBERTa-base and the output result can be found in:

[PLMs_reproduction/Flang_RoBERTa_base/](PLMs_reproduction/Flang_RoBERTa_base/)

In the original paper, RoBERTa-base trained on Combined-S data is the best-performing base size model, achieving test F1 of 0.6981. We use the same set of hyperparameters on the combined-S data, and get a test F1 of 0.6722.

The codes for reproduction of RoBERTa-base and the output result can be found in:

[PLMs_reproduction/RoBERTa_base/](PLMs_reproduction/RoBERTa_base/)

In the original paper, RoBERTa-large trained on Combined data is the overall best-performing model, achieving test F1 of 0.7171. We use the same set of hyperparameters on the combined data, and get a test F1 of 0.7204.

The codes for reproduction of RoBERTa-base and the output result can be found in:

[PLMs_reproduction/RoBERTa_large/](PLMs_reproduction/RoBERTa_large/)

### Search for additional hyperparameters on RoBERTa-large

In the original paper, the best-performing model on the sentence classification task is the RoBERTa-large model trained on combined data, with batch size 16, learning rate 1e-5, AdamW optimizer, and 20% of the training sentences randomly split into validation set.

We experiment with different batch sizes, learning rates, optimizer and percentage of validation data. For each set of hyperparameters, we experiment with 3 random seeds, and average the results.

The codes and experiment results can be found in:

[RoBERTa-large additional hyperparameters](Reproduction/)

### XLNet

XLNet, introduced by Yang et al. (2019), can achieve state-of-the-art results on several NLP benchmarks, surpassing BERT and other models on tasks like text classification, question answering, and sentiment analysis. Our study shows that it can perform closely to RoBERTa, and outperform RoBERTa-base on the PC-S dataset. 

We fine-tuned XLNet to do the hawkish-dovish sentence classification task; codes and experiment results can be found in:



### XLM-RoBERTa-base

XLM-RoBERTa-base proposed by Conneau et al. (2019) is an extension of the RoBERTa model, specifically designed to improve multilingual capabilities and performance across a wide range of languages. Although our study focused only on English sentences and observed that XLM-RoBERTa did not perform as well as RoBERTa, we recognize the importance of XLM-RoBERTa's multilingual capabilities. In a world where global interactions are the norm, XLM-RoBERTa's proficiency in various languages is invaluable, especially for understanding the monetary policies of central banks in non-English speaking countries.

We fine-tuned XLM-RoBERTa-base to do the hawkish-dovish sentence classification task; codes and experiment results can be found in:



## Implementing a trading strategy on the sentence classifier

Here is an application of the hawkish-dovish sentence classifier in constructing a trading strategy, same as that in the original paper. In this strategy ("our strategy"), a document-level hawkish tone measure is calculated on FOMC historical documents; this strategy will buy the QQQ index stocks when the FOMC shows a dovish tone, and short the QQQ index stocks when a hawkish tone is observed. The investment result is compared with the simple "buy and hold" strategy, where an investor simply buy the QQQ index at the beginning and hold till the end.

It can be seen in the output graphs that "our strategy" can outperform the "buy and hold strategy" to earn an extra return.
