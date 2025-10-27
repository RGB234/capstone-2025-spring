from datasets import load_dataset


## data loading ##

dataset_dir = "/data2/local_datasets/encoder/data"

# GPT-based dataset
# queries = load_dataset("json", data_files=f"{dataset_dir}/ft_test_queries.jsonl")[
#     "train"
# ]
# corpus = load_dataset("json", data_files=f"{dataset_dir}/ft_test_corpus.jsonl")["train"]
# qrels = load_dataset("json", data_files=f"{dataset_dir}/ft_test_qrels.jsonl")["train"]

# KLAID
# queries = load_dataset("json", data_files=f"{dataset_dir}/KLAID_test_queries.jsonl")[
#     "train"
# ]
# corpus = load_dataset("json", data_files=f"{dataset_dir}/KLAID_test_corpus.jsonl")[
#     "train"
# ]
# qrels = load_dataset("json", data_files=f"{dataset_dir}/KLAID_test_qrels.jsonl")[
#     "train"
# ]

# KLAID. K=10
queries = load_dataset(
    "json", data_files=f"{dataset_dir}/KLAID_test_queries_k10.jsonl"
)["train"]
corpus = load_dataset("json", data_files=f"{dataset_dir}/KLAID_test_corpus_k10.jsonl")[
    "train"
]
qrels = load_dataset("json", data_files=f"{dataset_dir}/KLAID_test_qrels_k10.jsonl")[
    "train"
]

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
        results[queries["id"][idx]] = {}  # len(results) == ceil(query_size/32)
        for score, index in zip(scores, indices):
            if index != -1:
                results[queries["id"][idx]][corpus["id"][index]] = float(score)

    return results


## Evaluation ##

from FlagEmbedding.abc.evaluation.utils import evaluate_metrics, evaluate_mrr
from FlagEmbedding import FlagModel


def evaluation(model_path: str, k_values: list[int] = [10, 50]):
    ## model loading ##
    model = FlagModel(
        model_path,
        devices=[0],
        use_fp16=False,
    )

    ## evaluation ##
    results = search(model, queries_text, corpus_text)

    eval_res = evaluate_metrics(qrels_dict, results, k_values)
    mrr = evaluate_mrr(qrels_dict, results, k_values)

    print(f"### {model_path} ###")
    for res in eval_res:
        print(res)
    print(mrr)


#
print("<< 파인튜닝 이전 >>")
evaluation(
    "dragonkue/bge-m3-ko",
)

print("<< 파인튜닝 이후 >>")

# BGE-M3
evaluation(
    "/data2/local_datasets/encoder/bgem3/ft_5ep",
)

evaluation(
    "/data2/local_datasets/encoder/bgem3/ft_10ep",
)

# Custom bge-m3 version 3
evaluation(
    "/data2/local_datasets/encoder/bgem3_custom/ft_5ep_v3",
)

evaluation(
    "/data2/local_datasets/encoder/bgem3_custom/ft_10ep_v3",
)

# Custom bge-m3 version 4
evaluation(
    "/data2/local_datasets/encoder/bgem3_custom/ft_5ep_v4",
)

evaluation(
    "/data2/local_datasets/encoder/bgem3_custom/ft_10ep_v4",
)
