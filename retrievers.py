
import os
from pathlib import Path
import bm25s
import pandas as pd
import numpy as np
import json 
import re
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API"))

Data_DIR = "parquet_data"
BM25_DIR = "indexes/BM25"
DENSE_DIR = "indexes/dense"
EMBEDDING_MODEL = "text-embedding-3-small"

# --------------------------------------------------------------
# BM25
# --------------------------------------------------------------

class BM25Retriever:
    def __init__(self) -> None:
        self._retriever = bm25s.BM25.load(str(BM25_DIR))
        self._doc_ids = (Path(BM25_DIR)/ "doc_ids.txt").read_text().splitlines()

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        TOKEN_PATTERN = r"""(?xu)
                        (?:[A-ZÁÀÉÈËÏÖÜÓÚ]{1,3}\.){2,}|
                        [A-ZÁÀÉÈËÏÖÜÓÚ]{2,}[0-9-]*|
                        [A-Za-zÀ-ÖØ-öø-ÿ]*['’][A-Za-zÀ-ÖØ-öø-ÿ]+|
                        [A-Za-zÀ-ÖØ-öø-ÿ]+(?:-[A-Za-zÀ-ÖØ-öø-ÿ0-9]+)+|
                        [A-Za-zÀ-ÖØ-öø-ÿ]+|
                        \d+(?:[./:-]\d+)*
                    """
        tokens = bm25s.tokenize([query], stopwords="dutch", token_pattern= TOKEN_PATTERN)
        indices, scores = self._retriever.retrieve(tokens, k=k)
        return [(self._doc_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0].tolist())]


class DenseRetriever:
    def __init__(self) -> None:
        corpus = pd.read_parquet(Path(Data_DIR)/"corpus_nl.parquet")
        self._doc_ids = corpus["id"].tolist()
        raw = np.load(Path(DENSE_DIR)/ "embeddings.npy")
        self._embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def _embed_query(self, query: str) -> np.ndarray:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self._embeddings @ self._embed_query(query)
        top_k = np.argsort(-scores)[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_k]


class HybridRetriever:
    def __init__(self, rrf_k):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.RRF_k = rrf_k
        
    def reciprocal_rank_fusion(self,rankings: list[list[str]]) -> list[tuple[str, float]]:
        k = self.RRF_k
        scores: dict[str, float] = defaultdict(float)
        for ranking in rankings:
            for rank, r_id in enumerate(ranking, start=1):
                scores[r_id] += 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: -x[1])

    def hybrid_candidates(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        bm25_ids = [int(doc_id) for doc_id, _ in self.bm25.search(query, k)]
        dense_ids = [int(doc_id) for doc_id, _ in self.dense.search(query,k)]
        return self.reciprocal_rank_fusion([bm25_ids, dense_ids])[:k]



class DocsRetriever:
    def __init__(self, search_results) -> None:
        self.search_results = search_results #list[tuple[document_id, score]]
        self.corpus = pd.read_parquet(Path(Data_DIR)/"corpus_nl.parquet") #"id" has a type of interger
        
    def documents(self) -> list[str]:
        document_ids = [int(art_id) for art_id, _ in self.search_results]
        document_texts = []

        for document_id in document_ids:
            text = self.corpus.loc[self.corpus["id"] == document_id, "article"].item()
            document_texts.append(text)
        ###could save as a dict {document_id: text}
        return document_texts
            
            
        
