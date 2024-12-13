from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class HybridRetriever:
    def __init__(self, vector_store, alpha=0.5, k=5):
        """
        Args:
            vector_store: Chroma vector store
            alpha: Trọng số cho vector search (1-alpha cho BM25)
            k: Số lượng documents trả về
        """
        self.vector_store = vector_store
        self.alpha = alpha
        self.k = k
        
        # Lấy cả documents và metadata
        vector_store_data = self.vector_store.get()
        all_docs = vector_store_data["documents"]
        all_metadatas = vector_store_data["metadatas"]
        
        # Tạo documents với metadata đầy đủ
        self.bm25_docs = []
        for text, metadata in zip(all_docs, all_metadatas):
            doc = Document(page_content=text, metadata=metadata)
            self.bm25_docs.append(doc)
            
        # Khởi tạo BM25 với documents đầy đủ metadata
        self.bm25_retriever = BM25Retriever.from_documents(self.bm25_docs)
        
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Chuẩn hóa scores về khoảng [0,1]"""
        if not scores:
            return scores
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def get_relevant_documents(
        self, query: str, callbacks: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Hybrid search kết hợp BM25 và vector similarity chạy song song"""
        
        # Chạy song song vector search và BM25 search
        with ThreadPoolExecutor() as executor:
            vector_future = executor.submit(
                self.vector_store.similarity_search_with_relevance_scores,
                query, self.k
            )
            bm25_future = executor.submit(
                self.bm25_retriever.invoke,
                query
            )
            
            # Lấy kết quả
            vector_docs = vector_future.result()
            bm25_docs = bm25_future.result()[:self.k]
        
        # Chuẩn hóa scores
        vector_scores = self._normalize_scores([score for _, score in vector_docs])
        vector_docs = [doc for doc, _ in vector_docs]
        bm25_scores = self._normalize_scores(
            [doc.metadata.get("score", 0) for doc in bm25_docs]
        )
        
        # Combine results
        doc_scores = {}
        
        # Add vector scores
        for doc, score in zip(vector_docs, vector_scores):
            doc_str = str(doc.page_content)
            if doc_str not in doc_scores:
                doc_scores[doc_str] = {"doc": doc, "score": 0}
            doc_scores[doc_str]["score"] += self.alpha * score
            
        # Add BM25 scores
        for doc, score in zip(bm25_docs, bm25_scores):
            doc_str = str(doc.page_content)
            if doc_str not in doc_scores:
                doc_scores[doc_str] = {"doc": doc, "score": 0}
            doc_scores[doc_str]["score"] += (1 - self.alpha) * score
            
        # Sort by combined scores
        sorted_docs = sorted(
            doc_scores.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )
        
        return [(item["doc"], item["score"]) for item in sorted_docs]

    def map(self):
        """Interface for RAG-Fusion with parallel processing"""
        def _map(queries):
            with ThreadPoolExecutor() as executor:
                # Thực thi song song các truy vấn
                future_results = [
                    executor.submit(self.get_relevant_documents, query) 
                    for query in queries
                ]
                # Thu thập kết quả
                results = [
                    [doc for doc, _ in future.result()[:self.k]]
                    for future in future_results
                ]
            return results
        return _map