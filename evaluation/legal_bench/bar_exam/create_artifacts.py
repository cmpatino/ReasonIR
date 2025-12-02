# Embedding datastore creation below
import argparse
import os

import faiss
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def _mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Average non-masked token embeddings to get a sentence embedding."""

    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    )
    sum_embeddings = (last_hidden_state * input_mask_expanded).sum(dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def encode_passages(
    model_name: str,
    texts: list[str],
    batch_size: int = 4,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding passages"):
        batch_texts = texts[start : start + batch_size]

        with torch.no_grad():
            outputs = model.encode(batch_texts, instruction="")

        embeddings.append(outputs.cpu())

    return torch.vstack(embeddings).numpy()


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def persist_artifacts(
    index: faiss.IndexFlatIP, ids: list[str], index_path: str, ids_path: str
) -> None:
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    with open(ids_path, "w") as f:
        for passage_id in ids:
            f.write(f"{passage_id}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create FAISS datastore for passages")
    parser.add_argument(
        "--model-name",
        type=str,
        default="reasonir/ReasonIR-8B",
        help="Hugging Face model id to use for embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of passages to encode per batch",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="data/passages/test.index",
        help="Where to store the FAISS index",
    )
    parser.add_argument(
        "--ids-path",
        type=str,
        default="data/passages/test_ids.txt",
        help="Where to store passage ids aligned with FAISS order",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    passages_local_path = hf_hub_download(
        repo_id="reglab/barexam_qa",
        filename="data/passages/test.tsv",
        repo_type="dataset",
        local_dir=".",
    )

    qa_local_path = hf_hub_download(
        repo_id="reglab/barexam_qa",
        filename="data/qa/test.csv",
        repo_type="dataset",
        local_dir=".",
    )

    passages_df = pd.read_csv(passages_local_path, sep="\t")[["idx", "text"]][:500]
    qa_df = pd.read_csv(qa_local_path)[["prompt", "question", "gold_idx"]]

    # Check all the gold passages are in the passages dataset
    merged_df = qa_df.merge(
        passages_df, left_on="gold_idx", right_on="idx", how="inner"
    )
    assert merged_df.shape[0] == qa_df.shape[0], (
        "Some gold passages are missing in the passages dataset"
    )

    del merged_df

    print(f"Number of QA pairs: {qa_df.shape[0]}")
    print(f"Number of Passages: {passages_df.shape[0]}")

    texts = passages_df["text"].tolist()
    embeddings = encode_passages(args.model_name, texts, batch_size=args.batch_size)
    index = build_faiss_index(embeddings)
    persist_artifacts(
        index, passages_df["idx"].tolist(), args.index_path, args.ids_path
    )


if __name__ == "__main__":
    main()
