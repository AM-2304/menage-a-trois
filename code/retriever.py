#!/usr/bin/env python3
"""
Enhanced Hybrid Retrieval Module

Three-way fusion: BM25+ · TF-IDF Cosine · LSA Embeddings
All deterministic, no external APIs or vector databases.

Improvements over basic BM25:
  - BM25+ scoring (fixes lower-bound bug for long documents)
  - Simple suffix stemming (payment/payments → payment)
  - Bigram indexing (phrase matching: "reset password")
  - Document chunking (long docs split into 300-token passages)
  - LSA dense embeddings via SVD (captures latent semantic relationships)
  - Three-way Reciprocal Rank Fusion for robust ranking
  - Product-hint boosting with validated paths
"""

import os
import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

# ─── Stopwords ────────────────────────────────────────────────────────────────

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

# ─── Simple suffix stemmer ────────────────────────────────────────────────────

# Words that should NOT be stemmed (domain-important terms)
_STEM_EXCEPTIONS = frozenset({
    'access', 'process', 'address', 'business', 'success', 'progress',
    'express', 'assessment', 'subscription', 'transaction', 'notification',
    'payment', 'refund', 'chargeback', 'invoice', 'password', 'candidate',
    'interview', 'proctoring', 'conversation', 'artifact', 'connector',
    'merchant', 'compliance', 'enterprise', 'permission', 'verification',
    'documentation', 'configuration', 'authentication', 'authorization',
})

# Order matters: check longer suffixes first
_SUFFIX_RULES = [
    ('ational', 'ate'), ('tional', 'tion'), ('fulness', 'ful'),
    ('iveness', 'ive'), ('ousness', 'ous'), ('ization', 'ize'),
    ('ations', 'ate'), ('ments', 'ment'), ('nings', 'ning'),
    ('ement', ''), ('ment', 'ment'), ('ness', ''),
    ('ling', ''), ('ally', 'al'), ('ying', 'y'),
    ('ies', 'y'), ('ing', ''), ('ely', ''),
    ('ous', ''), ('ive', ''), ('ful', ''),
    ('ed', ''), ('ly', ''), ('er', ''),
    ('ss', 'ss'),  # keep "access", "process"
    ('es', ''), ('s', ''),
]


def _stem(word: str) -> str:
    """Simple suffix-stripping stemmer. Handles common English suffixes."""
    if len(word) <= 4 or word in _STEM_EXCEPTIONS:
        return word
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix):
            result = word[:-len(suffix)] + replacement
            if len(result) >= 4:  # prevent over-stemming
                return result
    return word


def tokenize(text: str, stem: bool = True) -> List[str]:
    """Tokenize, remove stopwords, optionally stem."""
    raw = _TOKEN_RE.findall(text.lower())
    tokens = [t for t in raw if t not in STOPWORDS and len(t) > 1]
    if stem:
        tokens = [_stem(t) for t in tokens]
    return tokens


def make_bigrams(tokens: List[str]) -> List[str]:
    """Generate bigrams from token list for phrase matching."""
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


# ─── Optional numpy for LSA embeddings ────────────────────────────────────────

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ─── Corpus Index ─────────────────────────────────────────────────────────────

class CorpusIndex:
    """
    Enhanced retrieval index with three scoring methods:
      1. BM25+ (improved BM25 with lower-bound correction)
      2. TF-IDF cosine similarity
      3. LSA dense embeddings via SVD (if numpy available)
    Combined via Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        data_dir: str,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,       # BM25+ lower-bound parameter
        n_components: int = 128,  # LSA dimensions
        chunk_size: int = 300,    # tokens per chunk
        chunk_overlap: int = 100, # overlap between chunks
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.repo_root = os.path.dirname(self.data_dir)
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.n_components = n_components
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Document storage
        self.doc_paths: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_bigrams: List[List[str]] = []
        self.doc_contents: Dict[str, str] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.N: int = 0

        # BM25+ inverted index (unigrams + bigrams)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.df: Dict[str, int] = defaultdict(int)

        # TF-IDF
        self.idf: Dict[str, float] = {}
        self.tfidf_norms: List[float] = []

        # LSA embeddings (populated if numpy available)
        self.lsa_doc_embeds = None   # (N, n_components)
        self.lsa_vocab_list = None   # ordered vocab list
        self.lsa_vocab_idx = None    # term -> index
        self.lsa_components = None   # Vt[:n_components] for projecting queries

        # Document chunking
        self.chunk_tokens: List[List[str]] = []
        self.chunk_doc_ids: List[int] = []
        self.chunk_best_scores: Dict[int, float] = {}

        # Product mapping
        self.doc_products: List[str] = []

        self._build_index()

    def _build_index(self):
        """Full index build: scan → tokenize → BM25+ → TF-IDF → chunk → LSA."""

        # Phase 1: Scan and tokenize all documents
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
                tokens = tokenize(content, stem=True)
                if not tokens:
                    continue
                doc_id = len(self.doc_paths)
                self.doc_paths.append(rel_path)
                self.doc_tokens.append(tokens)
                bigrams = make_bigrams(tokens)
                self.doc_bigrams.append(bigrams)
                self.doc_contents[rel_path] = content
                self.doc_lengths.append(len(tokens))
                self.doc_products.append(self._infer_product(rel_path))

        self.N = len(self.doc_paths)
        if self.N == 0:
            return
        self.avg_doc_length = sum(self.doc_lengths) / self.N

        # Phase 2: Build BM25+ inverted index (unigrams + bigrams)
        for doc_id in range(self.N):
            # Unigrams
            tf_uni = Counter(self.doc_tokens[doc_id])
            for term, tf in tf_uni.items():
                self.inverted_index[term].append((doc_id, tf))
                self.df[term] += 1
            # Bigrams
            tf_bi = Counter(self.doc_bigrams[doc_id])
            for bigram, tf in tf_bi.items():
                self.inverted_index[bigram].append((doc_id, tf))
                self.df[bigram] += 1

        # Phase 3: Compute IDF
        for term, df_val in self.df.items():
            self.idf[term] = math.log((self.N - df_val + 0.5) / (df_val + 0.5) + 1.0)

        # Phase 4: TF-IDF L2 norms (unigrams only for cosine)
        for doc_id in range(self.N):
            tf_counter = Counter(self.doc_tokens[doc_id])
            norm_sq = sum(
                ((1 + math.log(tf)) * self.idf.get(t, 0)) ** 2
                for t, tf in tf_counter.items() if t in self.idf
            )
            self.tfidf_norms.append(math.sqrt(norm_sq) if norm_sq > 0 else 1.0)

        # Phase 5: Document chunking (for better passage-level matching)
        self._build_chunks()

        # Phase 6: LSA dense embeddings
        if HAS_NUMPY:
            self._build_lsa_embeddings()

    def _build_chunks(self):
        """Split long documents into overlapping chunks for finer-grained matching."""
        for doc_id in range(self.N):
            tokens = self.doc_tokens[doc_id]
            if len(tokens) <= self.chunk_size:
                self.chunk_tokens.append(tokens)
                self.chunk_doc_ids.append(doc_id)
            else:
                step = self.chunk_size - self.chunk_overlap
                for start in range(0, len(tokens), max(step, 1)):
                    chunk = tokens[start:start + self.chunk_size]
                    if len(chunk) >= 20:  # skip tiny trailing chunks
                        self.chunk_tokens.append(chunk)
                        self.chunk_doc_ids.append(doc_id)

    def _build_lsa_embeddings(self):
        """Build LSA embeddings: TF-IDF matrix → SVD → dense doc vectors."""
        # Build vocabulary (unigrams only, sorted for determinism)
        self.lsa_vocab_list = sorted(
            t for t in self.idf if '_' not in t  # exclude bigrams
        )
        self.lsa_vocab_idx = {t: i for i, t in enumerate(self.lsa_vocab_list)}
        V = len(self.lsa_vocab_list)
        n_comp = min(self.n_components, V - 1, self.N - 1)

        if n_comp < 2:
            return

        # Build TF-IDF matrix (N x V)
        tfidf_mat = np.zeros((self.N, V), dtype=np.float32)
        for doc_id in range(self.N):
            tf = Counter(self.doc_tokens[doc_id])
            for term, count in tf.items():
                if term in self.lsa_vocab_idx:
                    col = self.lsa_vocab_idx[term]
                    tfidf_mat[doc_id, col] = (1 + math.log(count)) * self.idf.get(term, 0)

        # L2 normalize rows
        norms = np.linalg.norm(tfidf_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tfidf_mat /= norms

        # Truncated SVD: keep top n_comp components
        U, S, Vt = np.linalg.svd(tfidf_mat, full_matrices=False)
        self.lsa_doc_embeds = U[:, :n_comp] * S[:n_comp]  # (N, n_comp)
        self.lsa_components = Vt[:n_comp]  # (n_comp, V) — for projecting queries

        # Normalize doc embeddings for cosine
        doc_norms = np.linalg.norm(self.lsa_doc_embeds, axis=1, keepdims=True)
        doc_norms[doc_norms == 0] = 1.0
        self.lsa_doc_embeds /= doc_norms

    @staticmethod
    def _infer_product(rel_path: str) -> str:
        parts = rel_path.lower().split('/')
        for p in ('claude', 'devplatform', 'visa'):
            if p in parts:
                return p
        return 'unknown'

    # ─── Scoring methods ──────────────────────────────────────────────────

    def _bm25plus_score(self, query_tokens: List[str], doc_id: int) -> float:
        """
        BM25+ scoring — fixes the lower-bound issue of standard BM25.
        Unlike BM25, a term that appears in a document always contributes
        a positive score (the delta parameter), even for very long documents.
        """
        score = 0.0
        doc_len = self.doc_lengths[doc_id]
        tf_counter = Counter(self.doc_tokens[doc_id])
        for qt in query_tokens:
            tf = tf_counter.get(qt, 0)
            if tf == 0 or qt not in self.idf:
                continue
            idf = self.idf[qt]
            norm_tf = tf / (1 - self.b + self.b * doc_len / self.avg_doc_length)
            bm25_component = idf * (norm_tf * (self.k1 + 1)) / (norm_tf + self.k1)
            # BM25+ addition: guaranteed minimum contribution
            score += bm25_component + self.delta * idf
        return score

    def _bm25plus_bigram_boost(self, query_tokens: List[str], doc_id: int) -> float:
        """Bonus score for bigram (phrase) matches."""
        q_bigrams = make_bigrams(query_tokens)
        if not q_bigrams:
            return 0.0
        doc_bi_counter = Counter(self.doc_bigrams[doc_id])
        boost = 0.0
        for qb in q_bigrams:
            if doc_bi_counter.get(qb, 0) > 0 and qb in self.idf:
                boost += self.idf[qb] * 0.5  # bigram boost weight
        return boost

    def _tfidf_cosine(self, query_tokens: List[str], doc_id: int) -> float:
        """TF-IDF cosine similarity between query and document."""
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

    def _lsa_cosine(self, query_tokens: List[str], doc_id: int) -> float:
        """LSA cosine similarity in dense embedding space."""
        if self.lsa_doc_embeds is None or self.lsa_components is None:
            return 0.0

        # Project query into LSA space
        q_vec = np.zeros(len(self.lsa_vocab_list), dtype=np.float32)
        tf = Counter(query_tokens)
        for term, count in tf.items():
            if term in self.lsa_vocab_idx:
                q_vec[self.lsa_vocab_idx[term]] = (1 + math.log(count)) * self.idf.get(term, 0)

        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return 0.0
        q_vec /= q_norm

        # Project into LSA space: q_lsa = q_vec @ Vt.T
        q_lsa = q_vec @ self.lsa_components.T  # (n_comp,)
        q_lsa_norm = np.linalg.norm(q_lsa)
        if q_lsa_norm == 0:
            return 0.0
        q_lsa /= q_lsa_norm

        # Cosine with doc embedding (already normalized)
        return float(np.dot(q_lsa, self.lsa_doc_embeds[doc_id]))

    def _chunk_best_score(self, query_tokens: List[str], doc_id: int) -> float:
        """Score document by its best-matching chunk (passage-level matching)."""
        best = 0.0
        query_set = set(query_tokens)
        for chunk_idx, cdoc_id in enumerate(self.chunk_doc_ids):
            if cdoc_id != doc_id:
                continue
            chunk_toks = self.chunk_tokens[chunk_idx]
            chunk_set = set(chunk_toks)
            overlap = len(query_set & chunk_set)
            if overlap == 0:
                continue
            # Simple overlap ratio score
            score = overlap / (len(query_set) + 0.5)
            best = max(best, score)
        return best

    # ─── Main search ──────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        product_hint: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float, str]]:
        """
        Search corpus using three-way RRF: BM25+ · TF-IDF · LSA.

        Returns list of (file_path, score, content_snippet) tuples.
        """
        if self.N == 0:
            return []

        query_tokens = tokenize(query, stem=True)
        if not query_tokens:
            return []

        query_bigrams = make_bigrams(query_tokens)

        # Find candidate docs (those with at least one query term or bigram)
        candidates = set()
        for qt in query_tokens + query_bigrams:
            if qt in self.inverted_index:
                for doc_id, _ in self.inverted_index[qt]:
                    candidates.add(doc_id)
        if not candidates:
            candidates = set(range(min(self.N, 100)))

        # Score with BM25+ (unigrams + bigram boost)
        bm25_scores = {}
        for d in candidates:
            bm25_scores[d] = self._bm25plus_score(query_tokens, d) + \
                             self._bm25plus_bigram_boost(query_tokens, d)

        # Score with TF-IDF cosine
        tfidf_scores = {d: self._tfidf_cosine(query_tokens, d) for d in candidates}

        # Score with LSA cosine (if available)
        lsa_scores = {}
        if HAS_NUMPY and self.lsa_doc_embeds is not None:
            for d in candidates:
                lsa_scores[d] = self._lsa_cosine(query_tokens, d)

        # Rank by each method
        bm25_ranked = sorted(candidates, key=lambda d: bm25_scores[d], reverse=True)
        tfidf_ranked = sorted(candidates, key=lambda d: tfidf_scores[d], reverse=True)

        # Reciprocal Rank Fusion (k=60) — 2-way or 3-way
        rrf = defaultdict(float)
        for rank, d in enumerate(bm25_ranked):
            rrf[d] += 1.0 / (60 + rank + 1)
        for rank, d in enumerate(tfidf_ranked):
            rrf[d] += 1.0 / (60 + rank + 1)

        if lsa_scores:
            lsa_ranked = sorted(candidates, key=lambda d: lsa_scores.get(d, 0), reverse=True)
            for rank, d in enumerate(lsa_ranked):
                rrf[d] += 1.0 / (60 + rank + 1)

        # Chunk-level boost: bonus for docs with high passage-level match
        for d in list(rrf.keys())[:50]:  # only top 50 to save time
            chunk_score = self._chunk_best_score(query_tokens, d)
            if chunk_score > 0:
                rrf[d] += chunk_score * 0.01  # small bonus

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
        return path in self.doc_contents

    def get_document(self, path: str) -> Optional[str]:
        return self.doc_contents.get(path)

    def get_all_paths(self) -> List[str]:
        return list(self.doc_paths)
