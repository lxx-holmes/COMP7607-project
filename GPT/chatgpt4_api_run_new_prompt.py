import openai,os,sys
openai.api_key = ""
# messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
# ]
import os

import pandas as pd
import numpy as np
from time import sleep

for seed in [5768, 78516, 944601]:
    for data_category in ["lab-manual-combine", "lab-manual-sp", "lab-manual-mm", "lab-manual-pc", "lab-manual-mm-split", "lab-manual-pc-split", "lab-manual-sp-split", "lab-manual-split-combine"]:

        # load training data
        test_data_path = "../training_data/test-and-training/test_data/" + data_category + "-test" + "-" + str(seed) + ".xlsx"
        data_df = pd.read_excel(test_data_path)


        sentences = data_df['sentence'].to_list()
        labels = data_df['label'].to_numpy()

        # exit(0)
        output_list = []
        for i in range(len(sentences)): 
            sen = sentences[i]
            message = "Given the FOMC's commitment to maximum employment, stable prices, and moderate long-term interest rates, consider how the sentence reflects these goals. The FOMC aims for transparency in its decisions, acknowledging the fluctuating nature of employment, inflation, and interest rates, and the influence of nonmonetary factors. It targets a 2% inflation rate over the long term, adjusting policy to manage shortfalls in employment and deviations in inflation. The Committee's decisions are informed by a range of indicators, balancing risks and long-term goals, with an annual review of its policy strategy. Example:  Input:  However, other participants noted that the continued subdued trend in wages was evidence of an absence of upward pressure on inflation from the current level of resource utilization. Output: DOVISH The sentence suggests that there is no significant wage-induced inflation pressure, which could imply that there is less need for tightening monetary policy. Your analysis should incorporate an understanding of the FOMC's principles, particularly how monetary policy actions, including the federal funds rate adjustments, play a role in achieving these objectives and responding to economic disturbances. The Sentence:" + sen
            # messages.append(
            #         {"role": "user", "content": message},
            # )
            messages = [
                    {"role": "user", "content": message},
            ]
            try:
                chat_completion = openai.ChatCompletion.create(
                        model="gpt-4-1106-preview",
                        messages=messages,
                        temperature=0.0,
                        max_tokens=1000
                )
            except Exception as e:
                print(e)
                i = i - 1
                sleep(10.0)

            answer = chat_completion.choices[0].message.content
            
            output_list.append([labels[i], sen, answer])
            sleep(1.0) 

            results = pd.DataFrame(output_list, columns=["true_label", "original_sent", "text_output"])

            results.to_csv(f'../llm_prompt_test_labels2/chatgpt_{data_category}_{seed}.csv', index=False)
