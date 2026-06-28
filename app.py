import os
import uuid
import re
from pathlib import Path
import streamlit as st

from agent.interview_assistant_service import InterviewAssistantService
from agent.agent_tools import get_city, get_weather
from rag.vector_store import VectorStoreService
from utils.path_tool import get_abs_path
from utils.user_history_store import load_user_state, save_user_state


st.set_page_config(page_title="基于RAG与Agent的多模态面试辅导助手(扁平版)", page_icon="💼", layout="wide")
st.title("💼 基于RAG与Agent的多模态面试辅导助手")


if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = f"guest_{uuid.uuid4().hex[:8]}"
if "interview_history" not in st.session_state:
    st.session_state.interview_history = []
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "interview_questions" not in st.session_state:
    st.session_state.interview_questions = []
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "interview_finished" not in st.session_state:
    st.session_state.interview_finished = False
if "interview_report" not in st.session_state:
    st.session_state.interview_report = ""
if "pending_interview_input" not in st.session_state:
    st.session_state.pending_interview_input = ""


def persist_state():
    save_user_state(
        st.session_state.current_user_id,
        {
            "interview_history": st.session_state.interview_history,
            "qa_history": st.session_state.qa_history,
            "interview_questions": st.session_state.interview_questions,
            "interview_started": st.session_state.interview_started,
            "interview_finished": st.session_state.interview_finished,
            "interview_report": st.session_state.interview_report,
        },
    )


UPLOAD_DIR = Path(get_abs_path("data/uploads"))


def sanitize_upload_filename(filename: str) -> str:
    name = Path(filename or "uploaded.txt").name
    stem = Path(name).stem.strip() or "uploaded"
    suffix = Path(name).suffix.lower()
    stem = re.sub(r"[^a-zA-Z0-9_\-\u4e00-\u9fff]+", "_", stem)
    return f"{stem}{suffix}"


def save_uploaded_knowledge_files(uploaded_files) -> list[str]:
    saved_paths = []
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for uploaded_file in uploaded_files or []:
        safe_name = sanitize_upload_filename(uploaded_file.name)
        if Path(safe_name).suffix.lower() not in [".txt", ".pdf"]:
            continue
        target_path = UPLOAD_DIR / safe_name
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(str(target_path))
    return saved_paths


def is_uploaded_knowledge_file(source_path: str) -> bool:
    try:
        source = Path(source_path).resolve()
        upload_dir = UPLOAD_DIR.resolve()
        return source == upload_dir or upload_dir in source.parents
    except Exception:
        return False


def render_knowledge_file_manager(show_toggle: bool = True):
    vector_service = VectorStoreService()
    knowledge_files = vector_service.list_knowledge_files()

    if show_toggle and not st.checkbox(f"显示文件管理（{len(knowledge_files)}个）", value=False):
        return

    if not knowledge_files:
        st.caption("暂无已登记的知识库文件。请先点击“加载/更新知识库”。")
        return

    st.caption("可查看已入库文件、chunk 数和 MD5。上传目录文件可直接删除文件与索引。")
    for index, item in enumerate(knowledge_files, start=1):
        source = item["source"]
        source_name = item["source_name"]
        md5_short = (item.get("md5") or "")[:8]
        chunk_count = item.get("chunk_count", 0)
        exists = item.get("exists", False)
        is_uploaded = is_uploaded_knowledge_file(source)

        status = "存在" if exists else "文件已缺失"
        st.markdown(f"**{index}. {source_name}**")
        st.caption(f"状态：{status}｜Chunks：{chunk_count}｜MD5：{md5_short}")
        st.caption(source)

        delete_label = "删除上传文件并移除索引" if is_uploaded else "仅移除索引"
        help_text = (
            "会删除 data/uploads/ 下的原文件，并从 ChromaDB 清理对应向量。"
            if is_uploaded
            else "只移除向量索引和登记信息；如果原文件仍在 data/ 下，下次加载知识库会重新入库。"
        )
        if st.button(
            delete_label,
            key=f"delete_knowledge_{index}_{md5_short}_{source_name}",
            help=help_text,
            use_container_width=True,
        ):
            result = vector_service.remove_knowledge_file(
                source,
                delete_physical_file=is_uploaded,
            )
            if is_uploaded:
                st.success(
                    f"已删除文件：{result['deleted_file']}；"
                    f"已删除向量：{result['deleted_vectors']} 条。"
                )
            else:
                st.success(f"已移除索引：{result['deleted_vectors']} 条。")
            st.rerun()

        if index != len(knowledge_files):
            st.divider()


def render_knowledge_page():
    st.subheader("知识库管理")
    st.caption("上传、更新、查看和删除知识库文件。知识文件会被切分、向量化并写入 ChromaDB。")

    upload_col, update_col = st.columns([2, 1])
    with upload_col:
        uploaded_knowledge_files = st.file_uploader(
            "上传知识文件（txt/pdf）",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            help="上传后文件会保存到 data/uploads/，该目录不会提交到 GitHub。",
        )
        if st.button("保存上传文件并更新知识库", use_container_width=True):
            if not uploaded_knowledge_files:
                st.warning("请先选择要上传的 txt 或 pdf 文件。")
            else:
                with st.spinner("正在保存文件并更新知识库..."):
                    saved_paths = save_uploaded_knowledge_files(uploaded_knowledge_files)
                    VectorStoreService().load_document()
                if saved_paths:
                    st.success(f"已保存 {len(saved_paths)} 个文件并更新知识库。")
                    for saved_path in saved_paths:
                        st.caption(saved_path)
                else:
                    st.warning("没有可保存的有效文件。")

    with update_col:
        st.markdown("**维护操作**")
        st.caption("文件未变化会自动跳过；文件变化会删除旧向量后重建。")
        if st.button("加载/更新知识库", use_container_width=True):
            with st.spinner("正在加载知识库..."):
                VectorStoreService().load_document()
            st.success("知识库加载完成")

    st.divider()
    render_knowledge_file_manager(show_toggle=False)


def render_chat_history(messages):
    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("role") == "assistant" and m.get("rag_trace"):
                render_rag_trace(m["rag_trace"])
            if m.get("role") == "assistant" and m.get("sources"):
                render_sources(m["sources"])


def stream_text(chunks, empty_message: str = "抱歉，我这次没有成功生成回答。请重试一次。") -> str:
    placeholder = st.empty()
    full_text = ""
    for chunk in chunks:
        if not chunk:
            continue
        full_text += str(chunk)
        placeholder.markdown(full_text + "▌")

    full_text = full_text.strip()
    if not full_text:
        full_text = empty_message
    placeholder.markdown(full_text)
    return full_text


def render_rag_trace(trace):
    if not trace:
        return
    status_labels = {
        "success": "Rerank 成功",
        "disabled": "未启用 Rerank",
        "skipped_no_docs": "无召回文档",
        "api_failed_degraded": "Rerank 接口失败，已降级",
        "empty_result_degraded": "Rerank 空结果，已降级",
        "exception_degraded": "Rerank 异常，已降级",
    }
    status = status_labels.get(trace.get("rerank_status", ""), trace.get("rerank_status", "未知"))
    with st.expander("查看本轮 RAG 检索信息", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("召回数量", trace.get("recall_count", 0))
        col2.metric("最终注入", trace.get("final_count", 0))
        col3.metric("向量检索", f"{trace.get('vector_elapsed_ms', 0)} ms")
        col4.metric("总检索", f"{trace.get('total_elapsed_ms', 0)} ms")
        st.caption(
            f"Rerank 状态：{status}｜"
            f"召回 Top-{trace.get('recall_k', '-') }｜"
            f"注入 Top-{trace.get('final_k', '-') }｜"
            f"重排耗时 {trace.get('rerank_elapsed_ms', 0)} ms"
        )


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"查看本轮参考资料（{len(sources)}条）", expanded=False):
        st.caption("参考资料按当前检索/重排后的顺序展示，排名越靠前通常与问题越相关。")
        for source in sources:
            index = source.get("rank") or source.get("index", "")
            source_name = source.get("source_name") or source.get("source", "未知来源")
            source_path = source.get("source", "未知来源")
            location = source.get("location", "")
            content = (source.get("content") or "").strip()
            content_length = source.get("content_length") or len(content)
            title = f"参考资料 {index}｜{source_name}"
            if location:
                title += f"｜{location}"
            st.markdown(f"**{title}**")
            st.caption(f"来源路径：{source_path}｜片段长度：{content_length} 字符")
            st.markdown(
                "> " + content.replace("\n", "\n> ")
                if content
                else "> 暂无可展示内容"
            )
            if index != len(sources):
                st.divider()


def generate_life_advice(weather_text: str) -> tuple[str, str]:
    if not weather_text:
        return "穿衣建议：暂无", "出行提醒：暂无"

    clothing = "穿衣建议：常规穿搭即可。"
    travel = "出行提醒：保持出行节奏，注意补水。"

    temp_match = re.search(r"气温\s*([\-]?\d+)", weather_text)
    temp = int(temp_match.group(1)) if temp_match else None
    lower_weather = weather_text.lower()

    if temp is not None:
        if temp <= 5:
            clothing = "面试穿衣建议：天气偏冷，建议厚外套/羽绒服+衬衫+加绒西装裤，整洁又干练，让面试官眼前一新！"
        elif temp <= 15:
            clothing = "面试穿衣建议：建议风衣+衬衫+厚西装裤，保暖不臃肿，你就是面试场上最靓的崽！"
        elif temp <= 26:
            clothing = "面试穿衣建议：温度舒适，建议穿白衬衫+西装裤，更得体哦~"
        else:
            clothing = "面试穿衣建议：天气较热，建议纯色短袖衬衫+垂感/直筒西装裤，显出你的重视！"

    if any(k in lower_weather for k in ["雨", "雷", "阵雨", "暴雨"]):
        travel = "出行提醒：可能降雨，建议带伞，注意路滑和交通安全。"
    elif any(k in lower_weather for k in ["雪", "冰"]):
        travel = "出行提醒：可能有雨雪结冰，建议减速慢行，注意防滑。"
    elif any(k in lower_weather for k in ["雾", "霾"]):
        travel = "出行提醒：能见度或空气质量一般，建议佩戴口罩并减少久留户外。"
    elif any(k in lower_weather for k in ["大风", "风"]):
        travel = "出行提醒：风力较大，注意高空坠物，骑行请减速。"

    return clothing, travel


def get_sidebar_weather_info(city_input: str = "") -> tuple[str, str]:
    manual_city = (city_input or "").strip()
    if manual_city:
        os.environ["CURRENT_USER_CITY"] = manual_city
        city = manual_city
    else:
        try:
            city = get_city.invoke({})
        except Exception:
            city = "未知城市"

    try:
        weather_text = get_weather.invoke({"city": city})
    except Exception:
        weather_text = "天气获取失败，请稍后重试。"
    return str(city), str(weather_text)


# 首次进入加载用户数据（只在当前用户未显式切换时执行）
if "user_state_loaded" not in st.session_state:
    loaded = load_user_state(st.session_state.current_user_id)
    st.session_state.interview_history = loaded.get("interview_history", [])
    st.session_state.qa_history = loaded.get("qa_history", [])
    st.session_state.interview_questions = loaded.get("interview_questions", [])
    st.session_state.interview_started = loaded.get("interview_started", False)
    st.session_state.interview_finished = loaded.get("interview_finished", False)
    st.session_state.interview_report = loaded.get("interview_report", "")
    st.session_state.user_state_loaded = True
    os.environ["CURRENT_USER_ID"] = st.session_state.current_user_id


service = InterviewAssistantService()

with st.sidebar:
    st.subheader("用户")
    user_col, switch_col = st.columns([2, 1])
    with user_col:
        user_id_input = st.text_input(
            "用户 ID",
            value=st.session_state.current_user_id,
            label_visibility="collapsed",
        )
    with switch_col:
        switch_user = st.button("切换", use_container_width=True)
    if switch_user:
        target_user_id = user_id_input.strip() or "guest"
        loaded = load_user_state(target_user_id)
        st.session_state.current_user_id = target_user_id
        st.session_state.interview_history = loaded.get("interview_history", [])
        st.session_state.qa_history = loaded.get("qa_history", [])
        st.session_state.interview_questions = loaded.get("interview_questions", [])
        st.session_state.interview_started = loaded.get("interview_started", False)
        st.session_state.interview_finished = loaded.get("interview_finished", False)
        st.session_state.interview_report = loaded.get("interview_report", "")
        os.environ["CURRENT_USER_ID"] = target_user_id
        st.success(f"已加载用户：{target_user_id}")
        st.rerun()

    st.subheader("天气与出行建议")
    city_input = st.text_input(
        "城市",
        value=os.getenv("CURRENT_USER_CITY", ""),
        placeholder="如：北京、杭州、上海",
        help="优先使用手动填写的城市；为空时尝试通过高德 IP 定位获取城市。",
    )
    city_name, weather_text = get_sidebar_weather_info(city_input)
    dress_advice, travel_advice = generate_life_advice(weather_text)
    st.caption(f"当前城市：{city_name}")
    st.caption(f"实时天气：{weather_text}")
    st.caption(dress_advice)
    st.caption(travel_advice)

    st.subheader("页面")
    mode = st.radio(
        "请选择页面",
        ["问答模式", "模拟面试", "知识库管理"],
        label_visibility="collapsed",
    )

if mode == "问答模式":
    st.subheader("问答模式")
    st.caption("用户提问，模型结合知识库与自身知识进行回答。")

    if st.button("清空问答历史", use_container_width=False):
        st.session_state.qa_history = []
        persist_state()
        st.rerun()

    render_chat_history(st.session_state.qa_history)

    question = st.chat_input("请输入你想问的问题...")
    if question:
        st.session_state.qa_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        result = service.qa_chat_with_sources_stream(question, st.session_state.qa_history)
        sources = result.get("sources", [])
        rag_trace = result.get("trace", {})
        with st.chat_message("assistant"):
            answer = stream_text(
                result.get("chunks", iter(())),
                "抱歉，我这次没有成功生成回答。请重试一次，或先点击“知识库管理”页面加载知识库后再提问。",
            )
            if rag_trace:
                render_rag_trace(rag_trace)
            if sources:
                render_sources(sources)

        st.session_state.qa_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "rag_trace": rag_trace,
            }
        )
        persist_state()
elif mode == "模拟面试":
    st.subheader("模拟面试模式")
    st.caption("模型将基于知识库模拟面试官提问。")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("开始/重置面试", use_container_width=True):
            st.session_state.interview_history = []
            st.session_state.interview_questions = []
            st.session_state.interview_report = ""
            st.session_state.interview_started = True
            st.session_state.interview_finished = False
            st.session_state.pending_interview_input = "请开始本次面试，先简单寒暄并提出第一个问题。"
            persist_state()
            st.rerun()

    with col2:
        if st.button("结束本次面试", use_container_width=True):
            st.session_state.interview_finished = True
            persist_state()
            st.rerun()

    with col3:
        st.write("当前状态：", "已结束" if st.session_state.interview_finished else "进行中")

    render_chat_history(st.session_state.interview_history)

    if st.session_state.pending_interview_input and st.session_state.interview_started and not st.session_state.interview_finished:
        pending_input = st.session_state.pending_interview_input
        st.session_state.pending_interview_input = ""
        with st.chat_message("assistant"):
            interviewer_reply = stream_text(service.interview_chat_stream(pending_input, st.session_state.interview_history))
        st.session_state.interview_history.append({"role": "assistant", "content": interviewer_reply})
        if "?" in interviewer_reply or "？" in interviewer_reply:
            st.session_state.interview_questions.append(interviewer_reply)
        persist_state()

    if st.session_state.interview_started and not st.session_state.interview_finished:
        user_reply = st.chat_input("请输入你的回答...")
        if user_reply:
            st.session_state.interview_history.append({"role": "user", "content": user_reply})
            with st.chat_message("user"):
                st.markdown(user_reply)
            with st.chat_message("assistant"):
                interviewer_reply = stream_text(service.interview_chat_stream(user_reply, st.session_state.interview_history))
            st.session_state.interview_history.append({"role": "assistant", "content": interviewer_reply})
            if "?" in interviewer_reply or "？" in interviewer_reply:
                st.session_state.interview_questions.append(interviewer_reply)
            persist_state()

    if st.session_state.interview_finished:
        want_report = st.checkbox("我希望生成本次面试报告", value=False)
        generated_report_now = False
        if want_report and st.button("生成面试报告", use_container_width=True):
            st.markdown("### 面试报告")
            with st.spinner("正在检索参考资料..."):
                report_chunks = service.generate_report_stream(
                    st.session_state.interview_history,
                    st.session_state.interview_questions,
                )
            st.session_state.interview_report = stream_text(report_chunks, "报告生成失败，请稍后重试。")
            generated_report_now = True
            persist_state()

        if st.session_state.interview_report and not generated_report_now:
            st.markdown("### 面试报告")
            st.markdown(st.session_state.interview_report)
else:
    render_knowledge_page()
