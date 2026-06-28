
"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from pathlib import Path
import time

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from rag.rerank_service import RerankService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.config_handler import rag_conf, chroma_conf


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.enable_rerank = bool(rag_conf.get("enable_rerank", False))
        self.recall_k = int(rag_conf.get("rerank_recall_k", 12))
        self.final_k = int(chroma_conf.get("k", 3))
        # 保持原有k语义：k=3代表最终给模型的文档数
        # 启用重排序时仅扩大初次召回数量，再重排回top-k
        retriever_k = self.recall_k if self.enable_rerank else self.final_k
        self.retriever = self.vector_store.get_retriever(k=retriever_k)
        self.rerank_service = RerankService() if self.enable_rerank else None
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template  | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        docs, _ = self.retriever_docs_with_trace(query)
        return docs

    def retriever_docs_with_trace(self, query: str) -> tuple[list[Document], dict]:
        started_at = time.perf_counter()
        vector_started_at = time.perf_counter()
        docs = self.retriever.invoke(query)
        vector_elapsed_ms = (time.perf_counter() - vector_started_at) * 1000

        trace = {
            "query": query,
            "rerank_enabled": self.enable_rerank,
            "recall_k": self.recall_k if self.enable_rerank else self.final_k,
            "final_k": self.final_k,
            "recall_count": len(docs),
            "final_count": len(docs),
            "vector_elapsed_ms": round(vector_elapsed_ms, 2),
            "rerank_elapsed_ms": 0.0,
            "total_elapsed_ms": 0.0,
            "rerank_status": "disabled",
        }

        final_docs = docs
        if self.enable_rerank and self.rerank_service:
            rerank_started_at = time.perf_counter()
            final_docs, rerank_status = self.rerank_service.rerank_with_status(query, docs)
            trace["rerank_elapsed_ms"] = round(
                (time.perf_counter() - rerank_started_at) * 1000, 2
            )
            trace["rerank_status"] = rerank_status

        trace["final_count"] = len(final_docs)
        trace["total_elapsed_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        return final_docs, trace

    @staticmethod
    def _build_context(context_docs: list[Document]) -> str:
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
        return context

    @staticmethod
    def format_source_docs(context_docs: list[Document]) -> list[dict]:
        """将检索文档转换成前端可展示的引用来源。"""
        sources = []
        for index, doc in enumerate(context_docs, start=1):
            metadata = doc.metadata or {}
            source_path = metadata.get("source") or metadata.get("file_path") or "未知来源"
            source_name = Path(str(source_path)).name if source_path else "未知来源"
            page = metadata.get("page")
            location = f"第 {int(page) + 1} 页" if isinstance(page, int) else ""
            content = (doc.page_content or "").strip()
            sources.append(
                {
                    "index": index,
                    "rank": index,
                    "source": str(source_path),
                    "source_name": source_name,
                    "location": location,
                    "content": content,
                    "content_length": len(content),
                    "metadata": metadata,
                }
            )
        return sources

    def rag_summarize_with_sources(self, query: str) -> dict:
        context_docs, trace = self.retriever_docs_with_trace(query)
        context = self._build_context(context_docs)
        answer = self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )
        return {
            "answer": answer,
            "sources": self.format_source_docs(context_docs),
            "trace": trace,
        }

    def rag_summarize_stream_with_sources(self, query: str) -> dict:
        context_docs, trace = self.retriever_docs_with_trace(query)
        context = self._build_context(context_docs)
        payload = {
            "input": query,
            "context": context,
        }

        def stream_answer():
            for chunk in self.chain.stream(payload):
                if chunk:
                    yield str(chunk)

        return {
            "chunks": stream_answer(),
            "sources": self.format_source_docs(context_docs),
            "trace": trace,
        }

    def rag_summarize(self, query: str) -> str:

        context_docs = self.retriever_docs(query)
        context = self._build_context(context_docs)

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("什么是线程？"))
