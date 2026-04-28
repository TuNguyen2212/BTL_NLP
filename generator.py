import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LegalGenerator:
    def __init__(self, api_key: str = None):
        # Collect all valid keys
        keys = []
        if api_key and api_key != "your_api_key_here":
            keys.append(api_key)
        for env_var in ["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY_3"]:
            k = os.getenv(env_var, "")
            if k and k != "your_api_key_here" and k not in keys:
                keys.append(k)
        if not keys:
            raise ValueError(
                "OPENROUTER_API_KEY chưa được cấu hình. "
                "Hãy tạo API key tại https://openrouter.ai/keys rồi điền vào file .env "
                "hoặc nhập trực tiếp trên giao diện."
            )
        self.clients = [
            OpenAI(base_url="https://openrouter.ai/api/v1", api_key=k)
            for k in keys
        ]
        print(f"[Generator] Loaded {len(self.clients)} API key(s)")
        self.models = [
            # Tier 1: Best quality for Vietnamese
            "google/gemma-4-31b-it:free",
            "google/gemma-3-12b-it:free",
            "google/gemma-3-27b-it:free",
            # Tier 2: Strong multilingual fallbacks
            "qwen/qwen3-coder:free",
            "openai/gpt-oss-120b:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            # Tier 3: Lightweight fallbacks
            "google/gemma-4-26b-a4b-it:free",
            "google/gemma-3-4b-it:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "nvidia/nemotron-nano-9b-v2:free",
        ]
        self.max_retries = 1
        self.base_delay = 2
        self.total_timeout = 45

    def _build_context(self, clauses: list[dict]) -> str:
        parts = []
        for c in clauses:
            entities_str = ""
            if c.get("entities"):
                ents = [f'{e["label"]}: {e["text"]}' for e in c["entities"]]
                entities_str = f'  Thực thể: {"; ".join(ents)}\n'

            parts.append(
                f'[{c["clause_id"]}] (Intent: {c.get("intent", "N/A")})\n'
                f'  "{c["text"]}"\n'
                f'{entities_str}'
            )
        return "\n".join(parts)

    def _build_prompt(self, query: str, clauses: list[dict]) -> str:
        context = self._build_context(clauses)

        return (
            "Bạn là trợ lý pháp lý chuyên phân tích hợp đồng tiếng Việt.\n"
            "\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. CHỈ trả lời dựa trên các mệnh đề hợp đồng được cung cấp bên dưới.\n"
            "2. Mỗi ý trong câu trả lời PHẢI kèm trích dẫn nguồn dạng [Cxxx].\n"
            "3. Nếu không có mệnh đề nào liên quan đến câu hỏi, trả lời: "
            '"Không tìm thấy thông tin liên quan trong hợp đồng."\n'
            "4. KHÔNG được suy luận hay thêm thông tin ngoài nội dung mệnh đề.\n"
            "5. Trả lời ngắn gọn, rõ ràng, đúng trọng tâm câu hỏi.\n"
            "\n"
            "--- MỆN ĐỀ HỢP ĐỒNG ---\n"
            f"{context}\n"
            "--- HẾT ---\n"
            "\n"
            f"CÂU HỎI: {query}\n"
            "\n"
            "TRẢ LỜI:"
        )

    def generate(self, query: str, clauses: list[dict]) -> dict:
        if not clauses:
            return {
                "answer": "Không tìm thấy mệnh đề nào liên quan trong hợp đồng.",
                "citations": [],
                "hallucination_check": {"passed": True, "reason": "Không có context"},
            }

        prompt = self._build_prompt(query, clauses)

        answer = self._call_llm(prompt)

        # Extract cited clause IDs from the answer
        cited_ids = list(dict.fromkeys(re.findall(r"C\d{3}", answer)))

        # Hallucination check
        hal_check = self._check_hallucination(answer, clauses, cited_ids)

        return {
            "answer": answer,
            "citations": cited_ids,
            "hallucination_check": hal_check,
        }

    def _call_llm(self, prompt: str) -> str:
        last_error = None
        start_time = time.time()
        for client_idx, client in enumerate(self.clients, 1):
            for model in self.models:
                # Check total timeout
                if time.time() - start_time > self.total_timeout:
                    raise TimeoutError(
                        "Tất cả model và API key đều đang bị rate limit. "
                        "Vui lòng đợi 1-2 phút rồi thử lại."
                    )
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    content = response.choices[0].message.content if response.choices else None
                    if not content:
                        print(f"[Generator] Empty response (key {client_idx}, {model}), trying next...")
                        continue
                    return content.strip()
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    if "429" in str(e) or "402" in str(e) or "rate" in err_str or "limit" in err_str or "quota" in err_str:
                        print(f"[Generator] Rate/quota limit (key {client_idx}, {model}), trying next...")
                        continue
                    else:
                        raise
        raise TimeoutError(
            "Tất cả model và API key đều đang bị rate limit. "
            "Vui lòng đợi 1-2 phút rồi thử lại."
        )

    def _check_hallucination(
        self, answer: str, clauses: list[dict], cited_ids: list[str]
    ) -> dict:
        context_ids = {c["clause_id"] for c in clauses}

        # Check 1: Are all cited IDs from the provided context?
        invalid_citations = [cid for cid in cited_ids if cid not in context_ids]
        if invalid_citations:
            return {
                "passed": False,
                "reason": f"Trích dẫn không hợp lệ (không có trong context): {invalid_citations}",
            }

        # Check 2: Does the answer have any citations at all?
        # If LLM correctly refuses (no relevant info), this is not hallucination
        refusal_phrases = [
            "không tìm thấy",
            "không có thông tin",
            "không liên quan",
            "không đủ thông tin",
        ]
        is_refusal = any(p in answer.lower() for p in refusal_phrases)

        if not cited_ids and not is_refusal:
            return {
                "passed": False,
                "reason": "Câu trả lời không có trích dẫn nguồn [Cxxx].",
            }

        if not cited_ids and is_refusal:
            return {
                "passed": True,
                "reason": "LLM từ chối trả lời do context không đủ liên quan — hợp lệ.",
            }

        # Check 3: Cross-check — verify key phrases from answer appear in source clauses
        source_texts = " ".join(
            c["text"] for c in clauses if c["clause_id"] in set(cited_ids)
        )

        # Extract monetary values and percentages from answer
        answer_numbers = set(re.findall(r"[\d.,]+\s*(?:%|VNĐ|đồng|triệu|tỷ|tháng|ngày|năm|m²)", answer))
        for num in answer_numbers:
            core = num.split()[0].rstrip("%")
            if core and core not in source_texts:
                return {
                    "passed": False,
                    "reason": f"Giá trị '{num}' trong câu trả lời không tìm thấy trong mệnh đề nguồn.",
                }

        return {"passed": True, "reason": "Câu trả lời bám sát nội dung mệnh đề."}
