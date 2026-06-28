from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import hashlib
import json
import os


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"],
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self, k: int | None = None):
        target_k = k if isinstance(k, int) and k > 0 else chroma_conf["k"]
        return self.vector_store.as_retriever(search_kwargs={"k": target_k})

    @staticmethod
    def _registry_path() -> str:
        return get_abs_path(chroma_conf["md5_hex_store"])

    @classmethod
    def load_registry(cls) -> dict:
        registry_path = cls._registry_path()
        if not os.path.exists(registry_path):
            return {"version": 2, "files": {}}

        with open(registry_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            return {"version": 2, "files": {}}

        try:
            registry = json.loads(content)
            if isinstance(registry, dict) and "files" in registry:
                return registry
        except json.JSONDecodeError:
            pass

        # 兼容旧版本：旧 md5.text 只有一行一个 md5，没有文件路径。
        return {"version": 2, "files": {}}

    @classmethod
    def save_registry(cls, registry: dict):
        registry_path = cls._registry_path()
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    @staticmethod
    def make_doc_id(source_path: str, md5_hex: str, chunk_index: int) -> str:
        raw = f"{source_path}|{md5_hex}|{chunk_index}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def delete_docs_by_source(self, source_path: str) -> int:
        try:
            result = self.vector_store.get(where={"source": source_path})
            ids = result.get("ids", []) if isinstance(result, dict) else []
            if ids:
                self.vector_store.delete(ids=ids)
            return len(ids)
        except Exception as e:
            logger.warning(f"[加载知识库]删除旧向量失败，source={source_path}, error={str(e)}")
            return 0

    def list_knowledge_files(self) -> list[dict]:
        registry = self.load_registry()
        files = registry.get("files", {})
        rows = []
        for source_path, info in files.items():
            rows.append(
                {
                    "source": source_path,
                    "source_name": info.get("source_name") or os.path.basename(source_path),
                    "md5": info.get("md5", ""),
                    "chunk_count": info.get("chunk_count", 0),
                    "exists": os.path.exists(source_path),
                }
            )
        return sorted(rows, key=lambda item: item["source_name"])

    def remove_knowledge_file(self, source_path: str, delete_physical_file: bool = False) -> dict:
        source_path = os.path.abspath(source_path)
        deleted_vectors = self.delete_docs_by_source(source_path)

        registry = self.load_registry()
        file_registry = registry.setdefault("files", {})
        existed_in_registry = source_path in file_registry
        file_registry.pop(source_path, None)
        self.save_registry(registry)

        deleted_file = False
        if delete_physical_file and os.path.exists(source_path):
            try:
                os.remove(source_path)
                deleted_file = True
            except Exception as e:
                logger.warning(f"[知识库管理]删除文件失败，source={source_path}, error={str(e)}")

        return {
            "deleted_vectors": deleted_vectors,
            "deleted_file": deleted_file,
            "existed_in_registry": existed_in_registry,
        }

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        registry = self.load_registry()
        file_registry = registry.setdefault("files", {})
        current_paths = {os.path.abspath(path) for path in allowed_files_path}

        # 清理已经从知识库目录删除的文件对应的旧向量
        for registered_path in list(file_registry.keys()):
            if registered_path not in current_paths:
                deleted_count = self.delete_docs_by_source(registered_path)
                file_registry.pop(registered_path, None)
                logger.info(f"[加载知识库]{registered_path}已不存在，删除旧向量{deleted_count}条")

        for path in allowed_files_path:
            path = os.path.abspath(path)
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)
            if not md5_hex:
                continue

            previous = file_registry.get(path)
            if previous and previous.get("md5") == md5_hex:
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                if previous:
                    deleted_count = self.delete_docs_by_source(path)
                    logger.info(f"[加载知识库]{path}内容已变化，删除旧向量{deleted_count}条")
                else:
                    # 兼容旧版本向量库：旧 md5.text 没有文件路径映射，首次升级时
                    # 仍尝试按 source 删除同文件旧向量，避免重复入库。
                    deleted_count = self.delete_docs_by_source(path)
                    if deleted_count:
                        logger.info(f"[加载知识库]{path}发现旧格式向量，删除{deleted_count}条后重建")

                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                ids = []
                for chunk_index, doc in enumerate(split_document):
                    doc.metadata = doc.metadata or {}
                    doc.metadata.update(
                        {
                            "source": path,
                            "source_name": os.path.basename(path),
                            "file_md5": md5_hex,
                            "chunk_index": chunk_index,
                        }
                    )
                    ids.append(self.make_doc_id(path, md5_hex, chunk_index))

                # 将内容存入向量库
                self.vector_store.add_documents(split_document, ids=ids)

                # 记录这个已经处理好的文件，避免下次重复加载；文件变化时可定位并删除旧向量
                file_registry[path] = {
                    "md5": md5_hex,
                    "chunk_count": len(split_document),
                    "source_name": os.path.basename(path),
                }
                self.save_registry(registry)

                logger.info(f"[加载知识库]{path} 内容加载成功，写入{len(split_document)}个片段")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue

        self.save_registry(registry)


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("线程")
    for r in res:
        print(r.page_content)
        print("-"*20)
