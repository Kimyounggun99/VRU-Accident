import json
from collections import defaultdict
import os
import string
import torch
import nltk
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from pycocoevalcap.spice.spice import Spice
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from comet import download_model, load_from_checkpoint
nltk.download('punkt')

def VQA_eval(args):

    input_path = args.load_path
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    
    json_files = [os.path.join(input_path, fname) for fname in os.listdir(input_path) if fname.endswith(".json")]
    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for video, categories in data.items():
            for category, content in categories.items():
                
                if content['pred'] == None:
                    pred= None
                else:
                    pred = content.get("pred", "").strip().upper()

                gt = content.get("GT", "").strip().upper()


                category_total[category] += 1
                if pred == gt:
                    category_correct[category] += 1


    save_path =args.save_path



    with open(save_path, "w", encoding="utf-8") as f:
        f.write("📊 Category-wise Accuracy:\n")
        print("📊 Category-wise Accuracy:")
        for category in category_total:
            correct = category_correct[category]
            total = category_total[category]
            acc = correct / total * 100
            line = f"- {category}: {correct}/{total} ({acc:.2f}%)"
            print(line)
            f.write(line + "\n")

    print(f"\n✅ Results was saved!: {save_path}")





 
def load_json_captions(filepath, field='gt'):
    """
    Load captions from a JSON file.
 
    Args:
        filepath (str): Path to JSON file.
        field (str): Field to extract ('gt' or 'pred').
 
    Returns:
        list: List of captions.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
 
    caption_list = [data[key][field] for key in data]
    print(f"Loaded {len(caption_list)} captions from {filepath}")
    return caption_list


def evaluate_spice(refs, hyps):
    """
    refs, hyps: two lists of 1000 strings each (reference & hypothesis captions)
    Returns: (overall_f1, individual_f1_list)
    """
    assert len(refs) == len(hyps) == 1000
 
    gts = {str(i): [refs[i]] for i in range(1000)}
    res = {str(i): [hyps[i]] for i in range(1000)}
 
    scorer = Spice()
    overall_f1, scores = scorer.compute_score(gts, res)
 
    # scores is a list of dicts per sample; we extract the F1 under 'All'
    f1_list = [score_dict["All"]["f"] for score_dict in scores]
 
    return overall_f1, f1_list

def evaluate_meteor(references, hypotheses):
    """
    Compute average METEOR score.
    """
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = [word for word in word_tokenize(ref) if word not in string.punctuation]
        hyp_tokens = [word for word in word_tokenize(hyp) if word not in string.punctuation]
        scores.append(meteor_score([ref_tokens], hyp_tokens))
    return sum(scores) / len(scores)
 
def evaluate_rouge(references, hypotheses):
    """
    Compute average ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    total_scores = {r: {'precision': 0, 'recall': 0, 'fmeasure': 0} for r in ['rouge1', 'rouge2', 'rougeL']}
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        for r_type, r_score in scores.items():
            total_scores[r_type]['precision'] += r_score.precision
            total_scores[r_type]['recall'] += r_score.recall
            total_scores[r_type]['fmeasure'] += r_score.fmeasure
 
    average_scores = {
        r: {
            'precision': total_scores[r]['precision'] / len(references),
            'recall': total_scores[r]['recall'] / len(references),
            'fmeasure': total_scores[r]['fmeasure'] / len(references),
        }
        for r in total_scores
    }
    return average_scores




def Dense_Captioning_eval(args):

    # Load COMET model
    comet_path = download_model("Unbabel/wmt22-comet-da")  # or another COMET model
    comet = load_from_checkpoint(comet_path)


    dataset_names = ["CAP_DATA", "DADA_2000", "DoTA", "MANUAL_DATA"]
    response_dir = f"./Model_Response/Dense_Captioning/{args.model}"
    gt_base_dir = "./MetaData"
    save_path = f"./results/Dense_Captioning/{args.model}_DenseCaption_Score.json"

    all_gen_captions = []
    all_gt_captions = []

    for dataset in dataset_names:
        pred_path = os.path.join(response_dir, f"{dataset}_Dense_Captioning_response.json")
        gt_path = os.path.join(gt_base_dir, dataset, f"{dataset}_Dense_Caption.json")

        print(f"🔹 Loading responses from {pred_path}")
        gen_captions = load_json_captions(pred_path, field='pred')

        print(f"🔹 Loading ground truths from {gt_path}")
        gt_captions = load_json_captions(gt_path, field='gt')

        assert len(gen_captions) == len(gt_captions), f" Mismatch in {dataset} caption counts!"

        all_gen_captions.extend(gen_captions)
        all_gt_captions.extend(gt_captions)
    
    print("Evaluating SPCIE...")
    avg_spice = evaluate_spice(all_gt_captions, all_gen_captions)

    print("Evaluating METEOR...")
    avg_meteor = evaluate_meteor(all_gt_captions, all_gen_captions)

    print("Evaluating ROUGE...")
    rouge_scores = evaluate_rouge(all_gt_captions, all_gen_captions)
    


    print("Evaluating COMET...")
    comet_data = [
    {"src": "", "mt": all_gen_captions[i], "ref": all_gt_captions[i]}
    for i in range(len(all_gt_captions))
    ]
    model_output = comet.predict(comet_data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
    comet_scores = model_output.scores
    avg_comet_score = sum(comet_scores) / len(comet_scores)
    
    avg_spice = avg_spice[0]

    result_dict = {
        "SPICE": round(avg_spice, 3),
        "METEOR": round(avg_meteor, 3),
        "COMET": round(avg_comet_score, 3),
        "ROUGE": {
            r_type.upper(): {
                "precision": round(scores["precision"], 3),
                "recall": round(scores["recall"], 3),
                "fmeasure": round(scores["fmeasure"], 3)
            } for r_type, scores in rouge_scores.items()
        },
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    print(f"\n Evaluation results saved to {save_path}")



 

