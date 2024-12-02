from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.load import dumps, loads
from typing import List
from langchain_core.documents import Document

class RAGFusionRetriever:
    def __init__(self, key_manager):
        self.key_manager = key_manager
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.key_manager.get_api_key(),
            temperature=0
        )
        
        # RAG-Fusion: Related - Hỗ trợ đa ngôn ngữ
        self.query_prompt = ChatPromptTemplate.from_template(
            """You are a helpful bilingual assistant that generates multiple search queries based on a single input query. 
            Generate queries in both Vietnamese and English regardless of the input language.

            Generate multiple search queries related to: {question}

            Output (4 queries, keep chemical terms unchanged in both languages):"""
        )

        # Tạo chain để generate queries
        self.generate_queries = (
            self.query_prompt 
            | self.llm 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )

    def reciprocal_rank_fusion(self, results: list[list], k=60):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
            and an optional parameter k used in the RRF formula """
        
        fused_scores = {}

        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                previous_score = fused_scores[doc_str]
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        return reranked_results

    def retrieve(self, question: str, retriever) -> List[tuple]:
        """Thực hiện RAG-Fusion retrieval với hỗ trợ đa ngôn ngữ"""
        # Cập nhật key mới trước khi gọi API
        self.llm.google_api_key = self.key_manager.get_api_key()
        
        # Sử dụng trực tiếp câu hỏi gốc
        retrieval_chain = (
            self.generate_queries
            | retriever.map()
            | self.reciprocal_rank_fusion
        )
        return retrieval_chain.invoke({"question": question}) 