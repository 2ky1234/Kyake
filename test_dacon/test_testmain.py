import argparse
import pandas as pd
from tqdm import tqdm
from rouge_metric import Rouge

class RougeScorer:
    def __init__(self):
        self.rouge_evaluator = Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=True,
            length_limit=1000,
            length_limit_type="words",
            use_tokenizer=True,
            apply_avg=True,
            apply_best=False,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
        )

    def compute_rouge(self, ref_df, hyp_df):    # ref_df, hyp_df : DataFrame
        #ref_df = pd.read_csv(ref_path)
        #hyp_df = pd.read_csv(hyp_path)
        hyp_df.iloc[:,1] = hyp_df.iloc[:,1].fillna(' ') # hyp_df의 두 번째 열만. na는 ' '으로 교체
        ids = ref_df['id']
        hyp_df = hyp_df[hyp_df['id'].isin(ids)]
        hyp_df.index = ref_df.index

        ref_df = ref_df.sort_values(by=["id"])
        hyp_df = hyp_df.sort_values(by=["id"])
        # print('ref_df["id"]', ref_df["id"])
        # print("ref_df['d']", ref_df["id"])
        ref_df["id"] = ref_df["id"].astype(int)
        hyp_df["id"] = hyp_df["id"].astype(int)

        hyps = [tuple(row) for row in hyp_df.values]
        refs = [tuple(row) for row in ref_df.values]

        reference_summaries = []
        generated_summaries = []

        for ref_tp, hyp_tp in zip(refs, hyps):
            ref_id, ref = ref_tp
            hyp_id, hyp = hyp_tp

            assert ref_id == hyp_id

            reference_summaries.append(ref)
            generated_summaries.append(hyp)

        scores = self.rouge_evaluator.get_scores(generated_summaries, reference_summaries)
        str_scores = self.format_rouge_scores(scores)
        #self.save_rouge_scores(str_scores)
        return str_scores

    def save_rouge_scores(self, str_scores):    # compute_rouge 함수의 return값을 txt파일로 저장하는 함수
        with open("rouge_scores.txt", "w") as output:
            output.write(str_scores)

    def format_rouge_scores(self, scores):      # score를 포멧에 맞춰 나열하는 함수?
    	return scores["rouge-1"]["f"], scores["rouge-2"]["f"], scores["rouge-l"]["f"],