from typing import Iterable, List
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from agent.agent_tools import rag_summarize
from model.factory import chat_model
from rag.rag_service import RagSummarizeService
from utils.prompt_loader import load_report_prompts, load_system_prompts, load_system_prompts2


class InterviewAssistantService:
    def __init__(self):
        self.rag_service = RagSummarizeService()
        self.tools = [rag_summarize]
        self.interview_executor = self._build_agent_executor(load_system_prompts())
        self.qa_executor = self._build_agent_executor(load_system_prompts2())
        self.report_chain = self._build_report_chain()

    @staticmethod
    def _to_agent_messages(history: List[dict]) -> List[dict]:
        lc_messages: List[dict] = []
        for message in history:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "user":
                lc_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                lc_messages.append({"role": "assistant", "content": content})
        return lc_messages

    def _build_agent_executor(self, system_prompt: str):
        return create_agent(
            model=chat_model,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=False,
        )

    @staticmethod
    def _build_report_chain():
        report_prompt = PromptTemplate.from_template(load_report_prompts())
        return report_prompt | chat_model | StrOutputParser()

    def interview_chat(self, user_input: str, history: List[dict]) -> str:
        messages = self._to_agent_messages(history)
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != user_input:
            messages.append({"role": "user", "content": user_input})
        response = self.interview_executor.invoke(
            {"messages": messages}
        )
        output = self._extract_ai_output(response)
        if output:
            return output
        # 兜底：避免agent返回结构变化导致空回复
        return self.rag_service.rag_summarize(user_input)

    def interview_chat_stream(self, user_input: str, history: List[dict]) -> Iterable[str]:
        messages = self._to_agent_messages(history)
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != user_input:
            messages.append({"role": "user", "content": user_input})

        try:
            yielded = False
            for event in self.interview_executor.stream(
                {"messages": messages},
                stream_mode="messages",
            ):
                text = self._stream_event_to_text(event)
                if text:
                    yielded = True
                    yield text
            if yielded:
                return
        except Exception:
            pass

        yield self.interview_chat(user_input, history)

    def qa_chat(self, user_input: str, history: List[dict]) -> str:
        messages = self._to_agent_messages(history)
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != user_input:
            messages.append({"role": "user", "content": user_input})
        response = self.qa_executor.invoke(
            {"messages": messages}
        )
        output = self._extract_ai_output(response)
        if output:
            return output
        # 兜底：保证问答模式至少返回RAG总结结果
        return self.rag_service.rag_summarize(user_input)

    def qa_chat_with_sources(self, user_input: str, history: List[dict]) -> dict:
        """问答模式专用：直接走 RAG，保证回答与展示来源来自同一次检索。"""
        rag_result = self.rag_service.rag_summarize_with_sources(user_input)
        return {
            "answer": rag_result.get("answer", ""),
            "sources": rag_result.get("sources", []),
            "trace": rag_result.get("trace", {}),
        }

    def qa_chat_with_sources_stream(self, user_input: str, history: List[dict]) -> dict:
        """问答模式专用：流式 RAG 回答，并附带同一次检索的来源。"""
        rag_result = self.rag_service.rag_summarize_stream_with_sources(user_input)
        return {
            "chunks": rag_result.get("chunks", iter(())),
            "sources": rag_result.get("sources", []),
            "trace": rag_result.get("trace", {}),
        }

    @staticmethod
    def _extract_ai_output(response: dict) -> str:
        direct_output = response.get("output")
        if isinstance(direct_output, str) and direct_output.strip():
            return direct_output.strip()

        messages = response.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return InterviewAssistantService._message_content_to_text(message.content)
            if isinstance(message, dict):
                role = message.get("role", "")
                if role == "assistant":
                    content = message.get("content", "")
                    return InterviewAssistantService._message_content_to_text(content)
        return ""

    @staticmethod
    def _message_content_to_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            text = content.get("text", "")
            if isinstance(text, str):
                return text
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "".join([part for part in text_parts if part])
        return ""

    @staticmethod
    def _stream_event_to_text(event) -> str:
        """兼容 LangChain/LangGraph 不同 stream 事件格式，尽量提取增量文本。"""
        if isinstance(event, tuple) and event:
            event = event[0]

        content = getattr(event, "content", None)
        if content:
            return InterviewAssistantService._message_content_to_text(content)

        if isinstance(event, dict):
            messages = event.get("messages")
            if isinstance(messages, list) and messages:
                return InterviewAssistantService._message_content_to_text(
                    getattr(messages[-1], "content", messages[-1].get("content", "") if isinstance(messages[-1], dict) else "")
                )
            for value in event.values():
                if isinstance(value, dict) and value.get("messages"):
                    nested_messages = value["messages"]
                    if nested_messages:
                        last = nested_messages[-1]
                        return InterviewAssistantService._message_content_to_text(
                            getattr(last, "content", last.get("content", "") if isinstance(last, dict) else "")
                        )
        return ""

    @staticmethod
    def _build_report_input(interview_history: List[dict], interview_questions: List[str]) -> str:
        rag_service = RagSummarizeService()
        full_log = []
        for message in interview_history:
            role = "候选人" if message["role"] == "user" else "面试官"
            full_log.append(f"{role}：{message['content']}")

        questions_text = "\n".join([f"{idx + 1}. {question}" for idx, question in enumerate(interview_questions)])
        question_query = "；".join(interview_questions) if interview_questions else "本次面试问题"

        docs = rag_service.retriever_docs(question_query)
        references = []
        for idx, doc in enumerate(docs, start=1):
            references.append(f"【参考资料{idx}】{doc.page_content}")

        return (
            f"【本次面试问题】\n{questions_text}\n\n"
            f"【完整对话记录】\n{chr(10).join(full_log)}\n\n"
            f"【知识库参考】\n{chr(10).join(references)}"
        )

    def generate_report(self, interview_history: List[dict], interview_questions: List[str]) -> str:
        interview_log = self._build_report_input(interview_history, interview_questions)

        return self.report_chain.invoke({"interview_log": interview_log})

    def generate_report_stream(self, interview_history: List[dict], interview_questions: List[str]) -> Iterable[str]:
        interview_log = self._build_report_input(interview_history, interview_questions)
        try:
            for chunk in self.report_chain.stream({"interview_log": interview_log}):
                if chunk:
                    yield str(chunk)
        except Exception:
            yield self.generate_report(interview_history, interview_questions)
