import os
import requests as _requests
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

load_dotenv()

CONTRACT_PATH = os.path.join(os.path.dirname(__file__), "input", "raw_contracts.txt")


st.set_page_config(
    page_title="RAG Chatbot – Hợp đồng pháp lý",
    layout="wide",
)


def _render_clause_cards(retrieved_clauses: list) -> str:
    clause_cards = []
    for c in retrieved_clauses:
        intent_html = f'<span class="intent-tag">{c.get("intent", "N/A")}</span>' if c.get("intent") else ""

        entities_html = ""
        if c.get("entities"):
            tags = [f'<span class="entity-tag">{e["label"]}: {e["text"]}</span>' for e in c["entities"]]
            entities_html = f'<div style="margin-top:4px">{"".join(tags)}</div>'

        contract_badge = ""
        if c.get("contract_name"):
            badge_color = "#4caf50" if c.get("contract_id") == "lease" else "#2196f3"
            contract_badge = (
                f'<span style="display:inline-block;background:{badge_color};color:#fff;'
                f'padding:1px 6px;border-radius:3px;font-size:0.75rem;margin-left:0.4rem">'
                f'{c["contract_name"]}</span>'
            )

        clause_cards.append(
            f'<div class="clause-card">'
            f'<span class="clause-id">{c["clause_id"]}</span>{intent_html}{contract_badge} '
            f'<small>(score: {c["score"]:.4f})</small>'
            f'<div style="margin-top:4px">{c["text"]}</div>'
            f'{entities_html}'
            f'</div>'
        )

    clauses_html = "\n".join(clause_cards)
    return (
        f'<details>'
        f'<summary style="cursor:pointer;padding:0.5rem 0;font-weight:600">'
        f'Xem {len(retrieved_clauses)} mệnh đề đã truy xuất</summary>'
        f'<div style="max-height:400px;overflow-y:auto;padding:0.3rem 0">{clauses_html}</div>'
        f'</details>'
    )


def _render_assistant_message(data: dict):
    hal_passed = data.get("hal_passed", True)
    answer = data.get("answer", "")
    hal_reason = data.get("hal_reason", "")
    citations = data.get("citations", [])
    retrieved_clauses = data.get("retrieved_clauses", [])

    if hal_passed:
        st.markdown(
            f'<div class="answer-box">{answer}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="warning-box">'
            f'<strong>Cảnh báo ảo giác:</strong> {hal_reason}<br><br>'
            f'{answer}</div>',
            unsafe_allow_html=True,
        )

    if hal_passed:
        st.success(f'Kiểm tra ảo giác: PASS — {hal_reason}', icon=":material/check_circle:")
    else:
        st.warning(f'Kiểm tra ảo giác: FAIL — {hal_reason}', icon=":material/warning:")

    if citations:
        st.info(f'Trích dẫn: {", ".join(citations)}', icon=":material/format_quote:")

    if retrieved_clauses:
        st.markdown(_render_clause_cards(retrieved_clauses), unsafe_allow_html=True)

st.markdown("""
<style>
    .main-title {
        text-align: center;
        padding: 0.5rem 0 0.2rem 0;
    }
    .subtitle {
        text-align: center;
        opacity: 0.7;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .answer-box {
        background: rgba(46, 125, 50, 0.1);
        border-left: 4px solid #2e7d32;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        line-height: 1.7;
    }
    .warning-box {
        background: rgba(249, 168, 37, 0.1);
        border-left: 4px solid #f9a825;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .clause-card {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
    }
    .clause-id {
        font-weight: 700;
        color: #42a5f5;
    }
    .intent-tag {
        display: inline-block;
        background: rgba(21, 101, 192, 0.15);
        color: #42a5f5;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .entity-tag {
        display: inline-block;
        background: rgba(198, 40, 40, 0.12);
        color: #ef5350;
        padding: 1px 6px;
        border-radius: 3px;
        font-size: 0.78rem;
        margin: 1px 2px;
    }

    @media (prefers-color-scheme: light) {
        .clause-id { color: #1565c0; }
        .intent-tag { color: #1565c0; background: rgba(21, 101, 192, 0.1); }
        .entity-tag { color: #c62828; background: rgba(198, 40, 40, 0.08); }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline(api_key: str = None):
    return RAGPipeline(api_key=api_key)


if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False


st.markdown('<h1 class="main-title">RAG Chatbot – Hợp đồng pháp lý</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Hỏi đáp tự động dựa trên nội dung hợp đồng tiếng Việt</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Cài đặt")
    env_key = os.getenv("OPENROUTER_API_KEY", "")
    has_env_key = bool(env_key) and env_key != "your_api_key_here"

    with st.expander("API Key" + (" (OK)" if has_env_key else " (Chưa có)"), expanded=not has_env_key):
        st.markdown(
            "Tạo key miễn phí tại [openrouter.ai/keys](https://openrouter.ai/keys) "
            "rồi dán vào ô bên dưới."
        )
        ui_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            placeholder="sk-or-v1-...",
            help="Để trống nếu đã cấu hình trong file .env",
        )
        active_key = ui_key.strip() if ui_key.strip() else (env_key if has_env_key else None)
        if active_key:
            st.success("Đang dùng key từ " + ("giao diện" if ui_key.strip() else ".env"), icon=":material/check_circle:")
            try:
                resp = _requests.get(
                    "https://openrouter.ai/api/v1/key",
                    headers={"Authorization": f"Bearer {active_key}"},
                    timeout=5,
                )
                if resp.status_code == 200:
                    kd = resp.json().get("data", {})
                    is_free = kd.get("is_free_tier", True)
                    daily = kd.get("usage_daily", 0)
                    daily_limit = 50 if is_free else 1000
                    remaining = max(0, daily_limit - daily)
                    tier_label = "Free" if is_free else "Paid"

                    st.markdown(
                        f"**{tier_label}** — "
                        f"Còn **{remaining}/{daily_limit}** req/ngày · "
                        f"Tối đa **20 req/phút**"
                    )
                    if remaining <= 5:
                        st.warning(
                            f"Gần hết quota ngày! Còn {remaining} request. "
                            "Quota reset lúc **00:00 UTC** (07:00 Việt Nam).",
                            icon=":material/warning:"
                        )
                elif resp.status_code == 401:
                    st.error("API key không hợp lệ.", icon=":material/error:")
                    active_key = None
            except Exception:
                pass  # Network error, skip status check
        else:
            st.warning("Chưa có API key. Vui lòng nhập ở trên hoặc cấu hình trong .env", icon=":material/warning:")

    if st.button("Tải lại Pipeline", help="Nhấn khi đổi API key hoặc gặp lỗi kết nối"):
        load_pipeline.clear()
        st.cache_resource.clear()
        st.session_state.pipeline_ready = False
        st.rerun()

    st.markdown("---")

    contract_options = {"all": "Tất cả hợp đồng", "lease": "HĐ Thuê mặt bằng", "labor": "HĐ Lao động"}
    selected_contract = st.selectbox(
        "Phạm vi tìm kiếm",
        options=list(contract_options.keys()),
        format_func=lambda x: contract_options[x],
        index=0,
        help="Chọn hợp đồng cụ thể để giới hạn phạm vi truy xuất, hoặc để mặc định 'Tất cả' để tìm trên cả hai hợp đồng.",
    )
    contract_filter = None if selected_contract == "all" else selected_contract

    top_k = st.slider("Số mệnh đề truy xuất (top-k)", min_value=1, max_value=20, value=10)

    st.divider()
    st.header("Hướng dẫn")
    st.markdown("""
    1. Chọn phạm vi truy xuất
    2. Nhập câu hỏi về hợp đồng
    3. Hệ thống truy xuất các mệnh đề liên quan
    4. LLM tạo câu trả lời có trích dẫn
    5. Kiểm tra ảo giác tự động
    """)

    st.divider()
    st.header("Hợp đồng gốc")
    show_contract = st.checkbox("Xem nội dung hợp đồng", value=False)

    if show_contract:
        try:
            with open(CONTRACT_PATH, "r", encoding="utf-8") as f:
                raw_text = f.read()
            # Split by double newline to detect separate contracts
            contracts = raw_text.strip().split("\n\n\n")
            if len(contracts) > 1:
                tabs = st.tabs([f"Hợp đồng {i+1}" for i in range(len(contracts))])
                for i, (tab, contract) in enumerate(zip(tabs, contracts)):
                    with tab:
                        st.text_area(
                            f"Nội dung hợp đồng {i+1}",
                            contract.strip(),
                            height=400,
                            disabled=True,
                            label_visibility="collapsed",
                        )
            else:
                st.text_area(
                    "Nội dung hợp đồng",
                    raw_text,
                    height=400,
                    disabled=True,
                    label_visibility="collapsed",
                )
        except FileNotFoundError:
            st.error("Không tìm thấy file hợp đồng gốc.")

    st.divider()
    st.header("Câu hỏi mẫu")
    examples = [
        "Giá thuê mặt bằng bao nhiêu?",
        "Khi nào phải thanh toán tiền thuê?",
        "Phạt chậm thanh toán như thế nào?",
        "Thời hạn thuê bao lâu?",
        "Mức lương cơ bản là bao nhiêu?",
        "Điều kiện chấm dứt hợp đồng?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["query_input"] = ex


if not active_key:
    st.info("Vui lòng nhập API key ở thanh bên trái để bắt đầu.", icon=":material/key:")
    st.stop()

if not st.session_state.pipeline_ready:
    loading_placeholder = st.empty()
    loading_placeholder.info(
        "**Đang khởi tạo hệ thống...**\n\n"
        "Lần đầu chạy cần tải model (~1-2 phút):\n"
        "- Embedding model (multilingual-e5-large ~1.2GB)\n"
        "- BM25 keyword index\n"
        "- Reranker model (~470MB)\n\n"
        "**Vui lòng chờ, KHÔNG tắt trang này!**",
        icon=":material/hourglass_empty:"
    )

try:
    with st.spinner("Đang tải pipeline... Vui lòng đợi."):
        pipeline = load_pipeline(active_key)
    st.session_state.pipeline_ready = True
    if 'loading_placeholder' in dir():
        loading_placeholder.empty()
except ValueError as e:
    st.error(str(e), icon=":material/error:")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and isinstance(msg.get("data"), dict):
            _render_assistant_message(msg["data"])
        elif msg["role"] == "assistant":
            st.markdown(msg.get("content", ""), unsafe_allow_html=True)
        else:
            st.write(msg["content"])


query = st.chat_input("Nhập câu hỏi về hợp đồng...")

if "query_input" in st.session_state:
    query = st.session_state.pop("query_input")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Đang truy xuất và phân tích..."):
                result = pipeline.answer(query, top_k=top_k, contract_id=contract_filter)
        except (TimeoutError, Exception) as e:
            err_msg = str(e)
            err_lower = err_msg.lower()
            if any(k in err_lower for k in ["rate limit", "limit", "quota"]) or "429" in err_msg or "402" in err_msg or isinstance(e, TimeoutError):
                st.error(
                    "**Tất cả model và API key đang bị giới hạn.**\n\n"
                    "- Free tier: tối đa **20 request/phút**, **50 request/ngày**\n"
                    "- Nếu bị limit phút, đợi **~60 giây** rồi thử lại\n"
                    "- Nếu hết quota ngày, reset lúc **00:00 UTC** (07:00 VN)\n"
                    "- Hoặc thêm API key mới vào `.env` rồi nhấn **Tải lại Pipeline**",
                    icon=":material/error:"
                )
            else:
                st.error(f"Lỗi: {err_msg}", icon=":material/error:")
            st.session_state.messages.pop()
            st.stop()

        msg_data = {
            "answer": result["answer"],
            "hal_passed": result["hallucination_check"]["passed"],
            "hal_reason": result["hallucination_check"]["reason"],
            "citations": result["citations"],
            "retrieved_clauses": result["retrieved_clauses"],
        }

        _render_assistant_message(msg_data)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "data": msg_data,
        })
