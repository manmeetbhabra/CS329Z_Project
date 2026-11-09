"""BM25 retriever on an inverted index (copied from reference_HW1_code/pipelines/bm25.py, HW1 Problem 3; only the import changed).

Scores follow the handout: k1 = 1.5, b = 0.75 and the Lucene IDF
ln((N - n_t + 0.5) / (n_t + 0.5) + 1). The index is built in one pass over
the corpus. A search reads only the postings of the query's terms, so its
cost grows with how common the query's terms are, not with the corpus size.
The plan, with the formula worked through, is in
code_planning/Part_1/Problem_3/.

Usage:
    index = BM25Index(["apple banana apple", "banana cherry"])
    index.search("banana", k=10)   # -> [(1, 0.2004...), (0, 0.1673...)]
"""

from __future__ import annotations

import heapq
import math
from collections import Counter

from agent.tokenizer import tokenize

K1 = 1.5   # term-frequency saturation: the tf factor never exceeds K1 + 1 = 2.5
B = 0.75   # length normalization: 0 ignores length, 1 scales fully with it


class BM25Index:
    """An inverted index over a list of documents, plus what BM25 needs to score them.

    Built once with ``BM25Index(docs)``, then searched any number of times
    with ``index.search(query, k)``.

    Attributes (values shown for the 5-document toy corpus in test_bm25.py):
        postings: term -> (doc_ids, tfs), two parallel lists. doc_ids are the
            documents that contain the term, in increasing order; tfs[i] is
            how often the term occurs in doc_ids[i].
            e.g. postings["cherry"] == ([1, 2, 3], [1, 3, 1])
        doc_len:  |d| for each doc_id, in tokens.    e.g. [3, 2, 4, 2, 4]
        norm:     K_d = k1 * (1 - b + b * |d| / avgdl) for each doc_id,
                  precomputed once.                   e.g. [1.5, 1.125, 1.875, 1.125, 1.875]
        n_docs:   N, the number of documents, blank ones included.  e.g. 5
        avgdl:    the mean of doc_len.                              e.g. 3.0
    """

    def __init__(self, docs: list[str]) -> None:
        """Index ``docs`` in a single pass.

        Args:
            docs: the corpus, one string per document. A document's id is
                its position: docs[0] is document 0. Blank strings are fine;
                they get length 0 and no postings, but still count in N.
                e.g. ["apple banana apple", "banana cherry", ""]
        """
        self.postings: dict[str, tuple[list[int], list[int]]] = {}
        self.doc_len: list[int] = []

        # The single pass: tokenize each document once, record its length,
        # and append its term counts to the postings.
        for doc_id, text in enumerate(docs):
            tokens = tokenize(text)          # "Banana, CHERRY!" -> ["banana", "cherry"]
            self.doc_len.append(len(tokens))
            self._add_document(doc_id, tokens)

        # Corpus totals, from the lengths alone (no second pass over the text).
        self.n_docs = len(self.doc_len)
        total = sum(self.doc_len)
        self.avgdl = total / self.n_docs if self.n_docs else 0.0

        # K_d depends only on the document, the same for every term and every
        # query. So compute it once here, not once per posting at query time.
        self.norm = [K1 * (1 - B + B * length / self.avgdl) for length in self.doc_len]

    def _add_document(self, doc_id: int, tokens: list[str]) -> None:
        """Append one document's term counts to the postings.

        Args:
            doc_id: the document's position in the corpus, e.g. 2.
            tokens: its tokens, e.g. ["cherry", "cherry", "cherry", "date"].

        After the call, for that example:
            postings["cherry"] has gained doc_id 2 with tf 3
            postings["date"]   has gained doc_id 2 with tf 1

        Documents are added in increasing doc_id order, so every postings
        list stays sorted without any sorting.
        """
        postings = self.postings  # local name: one lookup instead of one per term
        # Counter(tokens) -> {"cherry": 3, "date": 1}: one entry per distinct term.
        for term, tf in Counter(tokens).items():
            entry = postings.get(term)
            if entry is None:          # first document containing this term
                entry = postings[term] = ([], [])
            entry[0].append(doc_id)    # doc_ids
            entry[1].append(tf)        # tfs (same position)

    def idf(self, n_t: int) -> float:
        """Lucene IDF of a term that occurs in ``n_t`` of the ``n_docs`` documents.

        Args:
            n_t: number of documents containing the term, 1 <= n_t <= n_docs.
                 It is len(doc_ids) of the term's postings.

        Returns:
            ln((N - n_t + 0.5) / (n_t + 0.5) + 1), always > 0. Rarer -> larger.
        """
        return math.log((self.n_docs - n_t + 0.5) / (n_t + 0.5) + 1)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """The ``k`` best documents for ``query`` by BM25.

        Args:
            query: free text, tokenized exactly like the documents, e.g.
                "cherry date" or "BANANA... apple?". A term repeated in the
                query counts once. Terms found in no document are ignored.
            k: how many results to return at most, e.g. 10.

        Returns:
            A list of (doc_id, score) tuples, as (int, float): best score
            first, ties broken by the lower doc_id. Only documents sharing at
            least one term with the query, so it may have fewer than k items,
            or be [] (no tokens, or no known terms).
            e.g. search("cherry date", 2) -> [(2, 1.590503), (4, 0.761277)]
        """
        if k <= 0:
            return []
        # dict.fromkeys removes repeats but keeps query order:
        # "apple apple APPLE" -> ["apple", "apple", "apple"] -> {"apple"}
        # dict.fromkeys will crretae a dict with keys from the list of tokens
        # that come from the tokenize function. The list function will return
        # the keys of the dict as a list, which will be the distinct query terms.
        terms = list(dict.fromkeys(tokenize(query)))
        scores = self._score(terms)
        return self._top_k(scores, k)

    def _score(self, terms: list[str]) -> dict[int, float]:
        """BM25 score of every document that contains at least one of ``terms``.

        Args:
            terms: distinct query terms, e.g. ["cherry", "date"].

        Returns:
            doc_id -> summed score, for the matching documents only.

        This is the hot loop. It reads only the postings of ``terms``: a
        term in 3 of 200,000 documents costs 3 iterations, not 200,000.
        """
        scores: dict[int, float] = {}
        norm = self.norm          # local names: faster lookups inside the loop
        k1_plus_1 = K1 + 1
        for term in terms:
            entry = self.postings.get(term)
            if entry is None:
                continue          # e.g. "zucchini": in no document, adds nothing
            doc_ids, tfs = entry  # "cherry": [1, 2, 3], [1, 3, 1]
            idf = self.idf(len(doc_ids))  # once per term: n_t = 3 -> 0.538997
            for doc_id, tf in zip(doc_ids, tfs):
                # One term's contribution: IDF(t) * f(t,d) * (k1 + 1) / (f(t,d) + K_d).
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * k1_plus_1 / (tf + norm[doc_id])
        return scores

    @staticmethod
    def _top_k(scores: dict[int, float], k: int) -> list[tuple[int, float]]:
        """The ``k`` best (doc_id, score) pairs from ``scores``.

        Args:
            scores: doc_id -> score, e.g. {1: 0.634114, 2: 1.590503, 3: 0.634114, 4: 0.761277}.
            k: how many to keep, e.g. 2.

        Returns:
            Sorted by score descending, then doc_id ascending; (int, float) pairs.
            e.g. [(2, 1.590503), (4, 0.761277)]. With k = 4, (1, ...) comes before
            (3, ...) because they tie and 1 < 3.

        heapq.nsmallest on the key (-score, doc_id) keeps only k items while
        scanning
        """
        # Smallest (-score, doc_id) = highest score, then lowest doc_id on a tie..
        #
        # heapq.nsmallest(k, ..., key=...) returns the k items with the smallest key
        # values sorted smallest first. The key here is (-score, doc_id) which is tuple.
        # Tuples are compared lexicographically, so the first element is compared first, 
        # and if they are equal, the second element is compared. This means that the items 
        # with the highest score will have the smallest -score, and in case of a tie, 
        # the item with the smaller doc_id will come first. Note how the key function
        # uses item[1] for score and item[0] for doc_id, which are the values from the scores dictionary.
        #
        # top here is a list of tuples (doc_id, score) for the k best documents. 
        # The scores.items() returns an iterable of (doc_id, score) pairs from the 
        # scores dictionary.
        top = heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))

        # Return a list of (doc_id, score) tuples, where doc_id is an int and score is a float.
        return [(int(doc_id), float(score)) for doc_id, score in top]
