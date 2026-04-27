from retriever import ClauseRetriever
from src.evaluator import recall_at_k


def main():
    retriever = ClauseRetriever()

    test_set = [
        {"query": "giá thuê bao nhiêu", "expected": "C011"},
        {"query": "khi nào thanh toán tiền thuê", "expected": "C012"},
        {"query": "phạt chậm thanh toán như thế nào", "expected": "C013"},
        {"query": "thời hạn thuê bao lâu", "expected": "C006"}
    ]

    score = recall_at_k(
        retriever=retriever,
        test_set=test_set,
        k=3
    )

    print("\n📊 RETRIEVAL EVALUATION")
    print(f"Recall@3: {score:.4f}")


if __name__ == "__main__":
    main()