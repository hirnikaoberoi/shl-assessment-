"""BM25 lexical retrieval over the SHL catalog.

Deliberately not embedding-based: BM25 is deterministic, has zero external
API dependency (so retrieval never adds latency/cost/failure risk to a
30-second request budget), and needs no heavy ML runtime, which matters for
a free-tier deploy. Recall is boosted with a small hand-built synonym map
that bridges common recruiter vocabulary ("developer", "leadership") to the
vocabulary that actually appears in SHL's catalog text.
"""

import math
import re
from collections import Counter
from functools import lru_cache
from typing import List

from app.services.catalog import Assessment, load_catalog

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.]*")

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "for", "in", "on", "with",
    "is", "are", "be", "as", "at", "by", "we", "i", "need", "want", "looking",
    "hiring", "hire", "test", "tests", "assessment", "assessments", "please",
    "who", "that", "this", "our", "us", "will", "can", "would", "like",
    "someone", "candidate", "candidates", "role", "roles", "job", "jobs",
    "it", "also", "add",
}

# Bridges recruiter phrasing to SHL catalog vocabulary. Expansion terms are
# appended to the query, not injected into the documents, so scoring stays
# grounded in what the catalog actually says.
SYNONYMS = {
    "developer": ["programmer", "engineer", "coding", "software", "development"],
    "programmer": ["developer", "engineer", "coding"],
    "engineer": ["developer", "technical", "engineering"],
    "backend": ["server", "api", "database"],
    "frontend": ["ui", "web", "client"],
    "fullstack": ["frontend", "backend", "web"],
    "leadership": ["manager", "executive", "director", "leader", "management"],
    "manager": ["leadership", "supervisor", "management"],
    "executive": ["leadership", "director", "senior", "c-suite"],
    "stakeholder": ["communication", "collaboration", "interpersonal"],
    "junior": ["entry-level", "graduate", "early-career"],
    "senior": ["experienced", "professional", "advanced"],
    "graduate": ["entry-level", "junior", "campus"],
    "sales": ["customer", "commercial", "business development"],
    "customer": ["service", "support", "client"],
    "personality": ["behavior", "behaviour", "traits", "opq"],
    "cognitive": ["reasoning", "aptitude", "numerical", "verbal"],
    "coding": ["programming", "technical", "development"],
    "communication": ["verbal", "interpersonal", "stakeholder"],
    "analyst": ["analytical", "analysis", "data"],
    "administrative": ["admin", "clerical", "office"],
    "call": ["contact", "phone", "customer service"],
    "remote": ["virtual", "online"],
}


def tokenize(text: str) -> List[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def expand_query_tokens(tokens: List[str]) -> List[str]:
    expanded = list(tokens)
    for token in tokens:
        for extra in SYNONYMS.get(token, []):
            expanded.extend(tokenize(extra))
    return expanded


class BM25Index:
    def __init__(self, documents: List[Assessment], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_tokens: List[List[str]] = [tokenize(d.get("search_text", "")) for d in documents]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.doc_freqs: List[Counter] = [Counter(toks) for toks in self.doc_tokens]

        df = Counter()
        for toks in self.doc_tokens:
            for term in set(toks):
                df[term] += 1
        n_docs = len(documents)
        self.idf = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1) for term, freq in df.items()
        }

    def score(self, query_tokens: List[str], doc_index: int) -> float:
        freqs = self.doc_freqs[doc_index]
        dl = self.doc_len[doc_index]
        score = 0.0
        for term in query_tokens:
            if term not in freqs:
                continue
            idf = self.idf.get(term, 0.0)
            f = freqs[term]
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_doc_len or 1))
            score += idf * (f * (self.k1 + 1)) / (denom or 1)
        return score

    def search(self, query: str, top_k: int = 15) -> List[dict]:
        base_tokens = tokenize(query)
        query_tokens = expand_query_tokens(base_tokens)
        if not query_tokens:
            return []

        scored = []
        for i in range(len(self.documents)):
            s = self.score(query_tokens, i)
            if s > 0:
                scored.append((s, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            doc = dict(self.documents[idx])
            doc["_score"] = round(score, 4)
            results.append(doc)
        return results


@lru_cache(maxsize=1)
def get_index() -> BM25Index:
    return BM25Index(load_catalog())


def semantic_search(query: str, top_k: int = 15) -> List[dict]:
    return get_index().search(query, top_k=top_k)
