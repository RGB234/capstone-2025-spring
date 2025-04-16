from datasets import load_dataset

## data loading ##

dataset_dir = "/data2/local_datasets/bge-m3/ft_data"
queries = load_dataset("json", data_files=f"{dataset_dir}/test_queries.jsonl")["train"]
corpus = load_dataset("json", data_files=f"{dataset_dir}/test_corpus.jsonl")["train"]
qrels = load_dataset("json", data_files=f"{dataset_dir}/test_qrels.jsonl")["train"]

queries_text = queries["text"]
corpus_text = [text for sub in corpus["text"] for text in sub]

qrels_dict = {}
for line in qrels:
    if line["qid"] not in qrels_dict:
        qrels_dict[line["qid"]] = {}
    qrels_dict[line["qid"]][line["docid"]] = line["relevance"]


## Similarity search method ##

import faiss
import numpy as np
from tqdm import tqdm


def search(model, queries_text, corpus_text):

    queries_embeddings = model.encode_queries(queries_text)
    corpus_embeddings = model.encode_corpus(corpus_text)

    # create and store the embeddings in a Faiss index
    dim = corpus_embeddings.shape[-1]
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    index.train(corpus_embeddings)
    index.add(corpus_embeddings)

    query_size = len(queries_embeddings)

    all_scores = []
    all_indices = []

    # search top 100 answers for all the queries
    for i in tqdm(range(0, query_size, 32), desc="Searching"):
        j = min(i + 32, query_size)
        query_embedding = queries_embeddings[i:j]
        score, indice = index.search(query_embedding.astype(np.float32), k=100)
        all_scores.append(score)
        all_indices.append(indice)

    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)

    # store the results into the format for evaluation
    results = {}
    for idx, (scores, indices) in enumerate(zip(all_scores, all_indices)):
        results[queries["id"][idx]] = {}
        for score, index in zip(scores, indices):
            if index != -1:
                results[queries["id"][idx]][corpus["id"][index]] = float(score)

    return results


## Evaluation ##

from FlagEmbedding.abc.evaluation.utils import evaluate_metrics, evaluate_mrr
from FlagEmbedding import FlagModel

k_values = [10, 100]

## Raw model

raw_name = "BAAI/bge-m3"
finetuned_path = "/data2/local_datasets/bge-m3/finetuned_bge-m3"

raw_model = FlagModel(
    raw_name,
    # query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    devices=[0],
    use_fp16=False,
)

results = search(raw_model, queries_text, corpus_text)

eval_res = evaluate_metrics(qrels_dict, results, k_values)
mrr = evaluate_mrr(qrels_dict, results, k_values)

for res in eval_res:
    print(res)
print(mrr)

## Fine tuning model

ft_model = FlagModel(
    finetuned_path,
    # query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    devices=[0],
    use_fp16=False,
)

results = search(ft_model, queries_text, corpus_text)

eval_res = evaluate_metrics(qrels_dict, results, k_values)
mrr = evaluate_mrr(qrels_dict, results, k_values)

for res in eval_res:
    print(res)
print(mrr)
