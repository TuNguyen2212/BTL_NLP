from retriever import ClauseRetriever
from generator import LegalGenerator


class RAGPipeline:
    def __init__(self, api_key: str = None):
        print("[RAG] Khởi tạo pipeline...")
        self.retriever = ClauseRetriever()
        self.generator = LegalGenerator(api_key=api_key)
        print("[RAG] Pipeline sẵn sàng.")

    def answer(self, query: str, top_k: int = 5, contract_id: str = None) -> dict:
        clauses = self.retriever.search(query, top_k=top_k, contract_id=contract_id)
        result = self.generator.generate(query, clauses)
        result["retrieved_clauses"] = clauses

        if not result["hallucination_check"]["passed"]:
            result["answer"] = (
                "CẢNH BÁO: Phát hiện khả năng ảo giác.\n"
                f'Lý do: {result["hallucination_check"]["reason"]}\n\n'
                f'Câu trả lời gốc (chưa xác minh):\n{result["answer"]}'
            )

        return result


def main():
    pipeline = RAGPipeline()

    print("\n[RAG] CHATBOT - Hợp đồng pháp lý")
    print("Gõ 'exit' để thoát.\n")

    while True:
        query = input("Câu hỏi: ")
        if query.strip().lower() == "exit":
            print("Tạm biệt!")
            break

        result = pipeline.answer(query)

        print(f"\nTrả lời:\n{result['answer']}")
        print(f"\nTrích dẫn: {result['citations']}")
        print(f"Hallucination check: {'PASS' if result['hallucination_check']['passed'] else 'FAIL'}")
        print(f"  {result['hallucination_check']['reason']}")
        print("-" * 80)
        print()


if __name__ == "__main__":
    main()
