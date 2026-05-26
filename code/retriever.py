#!/usr/bin/env python3
"""
Hybrid BM25 + TF-IDF Retrieval Module

Deterministic, stdlib-based retrieval over the support corpus.
No external vector databases or embedding APIs required.

Architecture:
- BM25 scoring (k1=1.5, b=0.75) for term-frequency based ranking
- TF-IDF cosine similarity for semantic matching
- Reciprocal Rank Fusion (RRF) to combine both rankings
- Product-hint boosting for documents matching inferred product
- All file paths validated against actual filesystem
"""

import os
import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "that", "this", "was", "are",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "not", "no", "so", "if",
    "as", "its", "my", "your", "we", "they", "he", "she", "i", "me",
    "him", "her", "us", "them", "what", "which", "who", "when", "where",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "than", "too", "very", "just", "about", "above",
    "after", "again", "also", "am", "any", "because", "been", "before",
    "being", "below", "between", "came", "come", "doing", "down", "during",
    "further", "get", "got", "having", "here", "herself", "himself",
    "itself", "made", "make", "many", "mine", "much", "must", "myself",
    "never", "nor", "now", "off", "often", "once", "only", "our", "ours",
    "ourselves", "out", "over", "own", "said", "same", "see", "shall",
    "since", "still", "take", "tell", "their", "themselves", "then",
    "there", "these", "thing", "things", "those", "through", "thus",
    "together", "under", "until", "unto", "up", "upon", "use", "used",
    "using", "want", "way", "well", "were", "while", "whom", "why",
    "without", "yet", "you", "yours", "yourself", "yourselves",
})

_TOKEN_RE = re.compile(r'[a-z0-9]+')


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words, removing stopwords and short tokens."""
    return [t for t in _TOKEN_RE.findall(text.lower())
            if t not in STOPWORDS and len(t) > 1]


class CorpusIndex:
    """Hybrid BM25 + TF-IDF index over the support corpus."""

    def __init__(self, data_dir: str, k1: float = 1.5, b: float = 0.75):
        self.data_dir = os.path.abspath(data_dir)
        self.repo_root = os.path.dirname(self.data_dir)
        self.k1 = k1
        self.b = b

        self.doc_paths: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_contents: Dict[str, str] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.N: int = 0

        # BM25
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.df: Dict[str, int] = defaultdict(int)

        # TF-IDF
        self.idf: Dict[str, float] = {}
        self.tfidf_norms: List[float] = []

        # Product mapping
        self.doc_products: List[str] = []

        self._build_index()

    def _build_index(self):
        """Scan corpus, tokenize, and build BM25 + TF-IDF indices."""
        for root, _, files in os.walk(self.data_dir):
            for fname in sorted(files):
                if not fname.endswith('.md'):
                    continue
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, self.repo_root).replace('\\', '/')

                try:
                    with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                except Exception:
                    continue

                tokens = tokenize(content)
                if not tokens:
                    continue

                doc_id = len(self.doc_paths)
                self.doc_paths.append(rel_path)
                self.doc_tokens.append(tokens)
                self.doc_contents[rel_path] = content
                self.doc_lengths.append(len(tokens))
                self.doc_products.append(self._infer_product(rel_path))

                tf_counter = Counter(tokens)
                for term, tf in tf_counter.items():
                    self.inverted_index[term].append((doc_id, tf))
                    self.df[term] += 1

        self.N = len(self.doc_paths)
        if self.N == 0:
            return

        self.avg_doc_length = sum(self.doc_lengths) / self.N

        for term, df_val in self.df.items():
            self.idf[term] = math.log((self.N - df_val + 0.5) / (df_val + 0.5) + 1.0)

        for doc_id in range(self.N):
            tf_counter = Counter(self.doc_tokens[doc_id])
            norm_sq = sum(
                ((1 + math.log(tf)) * self.idf.get(term, 0)) ** 2
                for term, tf in tf_counter.items() if term in self.idf
            )
            self.tfidf_norms.append(math.sqrt(norm_sq) if norm_sq > 0 else 1.0)

    @staticmethod
    def _infer_product(rel_path: str) -> str:
        parts = rel_path.lower().split('/')
        for p in ('claude', 'devplatform', 'visa'):
            if p in parts:
                return p
        return 'unknown'

    def _bm25_score(self, query_tokens: List[str], doc_id: int) -> float:
        score = 0.0
        doc_len = self.doc_lengths[doc_id]
        tf_counter = Counter(self.doc_tokens[doc_id])
        for qt in query_tokens:
            tf = tf_counter.get(qt, 0)
            if tf == 0 or qt not in self.idf:
                continue
            idf = self.idf[qt]
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score += idf * num / den
        return score

    def _tfidf_cosine(self, query_tokens: List[str], doc_id: int) -> float:
        query_tf = Counter(query_tokens)
        doc_tf = Counter(self.doc_tokens[doc_id])
        q_norm_sq = 0.0
        dot = 0.0
        for qt, qtf in query_tf.items():
            if qt not in self.idf:
                continue
            q_w = (1 + math.log(qtf)) * self.idf[qt]
            q_norm_sq += q_w ** 2
            d_tf = doc_tf.get(qt, 0)
            if d_tf > 0:
                d_w = (1 + math.log(d_tf)) * self.idf[qt]
                dot += q_w * d_w
        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0
        d_norm = self.tfidf_norms[doc_id]
        return dot / (q_norm * d_norm) if q_norm * d_norm > 0 else 0.0

    def search(
        self,
        query: str,
        product_hint: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float, str]]:
        """
        Search corpus using hybrid BM25 + TF-IDF with Reciprocal Rank Fusion.

        Returns list of (file_path, score, content_snippet) tuples.
        """
        if self.N == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Candidate docs: those with at least one query term
        candidates = set()
        for qt in query_tokens:
            if qt in self.inverted_index:
                for doc_id, _ in self.inverted_index[qt]:
                    candidates.add(doc_id)

        if not candidates:
            candidates = set(range(min(self.N, 100)))

        # Score with both methods
        bm25 = {d: self._bm25_score(query_tokens, d) for d in candidates}
        tfidf = {d: self._tfidf_cosine(query_tokens, d) for d in candidates}

        # Rank by each
        bm25_ranked = sorted(candidates, key=lambda d: bm25[d], reverse=True)
        tfidf_ranked = sorted(candidates, key=lambda d: tfidf[d], reverse=True)

        # Reciprocal Rank Fusion (k=60)
        rrf = defaultdict(float)
        for rank, d in enumerate(bm25_ranked):
            rrf[d] += 1.0 / (60 + rank + 1)
        for rank, d in enumerate(tfidf_ranked):
            rrf[d] += 1.0 / (60 + rank + 1)

        # Product hint boost
        if product_hint:
            ph = product_hint.lower().strip()
            for d in rrf:
                if self.doc_products[d] == ph:
                    rrf[d] *= 1.5

        ranked = sorted(rrf, key=lambda d: rrf[d], reverse=True)

        results = []
        for d in ranked[:top_k]:
            path = self.doc_paths[d]
            content = self.doc_contents.get(path, "")
            snippet = content[:500].strip()
            results.append((path, rrf[d], snippet))
        return results

    def path_exists(self, path: str) -> bool:
        """Check if a document path exists in the corpus."""
        return path in self.doc_contents

    def get_document(self, path: str) -> Optional[str]:
        """Get full document content by path."""
        return self.doc_contents.get(path)

    def get_all_paths(self) -> List[str]:
        """Get all document paths in the corpus."""
        return list(self.doc_paths)
