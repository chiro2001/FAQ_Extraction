import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge  # pip install rouge


def cal_metric(data):
    rouge = Rouge()
    smooth = SmoothingFunction().method1
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    total = 0
    for i in range(len(data)):
        total += 1
        pred_q = data.loc[i, "predict_question"]
        true_q = data.loc[i, "truth_question"]
        pred_q = " ".join(pred_q)
        true_q = " ".join(true_q)
        scores = rouge.get_scores(hyps=pred_q, refs=true_q)
        rouge_1 += scores[0]['rouge-1']['f']
        rouge_2 += scores[0]['rouge-2']['f']
        rouge_l += scores[0]['rouge-l']['f']
        bleu += sentence_bleu(
            references=[true_q.split(' ')],
            hypothesis=pred_q.split(' '),
            smoothing_function=smooth
        )
        #break
    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    bleu /= total

    result = {
                'rouge-1': rouge_1,
                'rouge-2': rouge_2,
                'rouge-l': rouge_l,
                'bleu': bleu,
            }
    return result


if __name__ == '__main__':

    print("----- new data new model beam search------")
    tricks_data = pd.read_csv("./prediction_result/test_predict_sougou-zhaoshang-fold_1-5_least_data_least_model.csv")
    tricks_result = cal_metric(tricks_data)
    print(tricks_result)

