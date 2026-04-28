from retriever import ClauseRetriever
from src.evaluator import recall_at_k

TEST_SET = [
    {"query": "giá thuê mặt bằng bao nhiêu tiền một tháng", "expected": "C011"},
    {"query": "khi nào phải thanh toán tiền thuê", "expected": "C012"},
    {"query": "phạt chậm thanh toán tiền thuê như thế nào", "expected": "C013"},
    {"query": "thời hạn thuê mặt bằng bao lâu", "expected": "C006"},
    {"query": "tiền đặt cọc thuê mặt bằng là bao nhiêu", "expected": "C016"},
    {"query": "điều kiện chấm dứt hợp đồng thuê trước hạn", "expected": "C042"},
    {"query": "Bên B có được phép sửa chữa cải tạo mặt bằng không", "expected": "C037"},
    {"query": "mức lương cơ bản của người lao động", "expected": "C065"},
    {"query": "nghĩa vụ bảo mật thông tin sau khi nghỉ việc", "expected": "C098"},
    {"query": "điều khoản cấm cạnh tranh sau khi nghỉ việc", "expected": "C111"},
]


def main():
    retriever = ClauseRetriever()

    for k in [3, 5]:
        score = recall_at_k(retriever=retriever, test_set=TEST_SET, k=k)
        print(f"Recall@{k}: {score:.4f} ({int(score * len(TEST_SET))}/{len(TEST_SET)})")

    print()
    print("-" * 80)
    for item in TEST_SET:
        results = retriever.search(item["query"], top_k=5)
        retrieved_ids = [r["clause_id"] for r in results]
        hit = "[HIT]" if item["expected"] in retrieved_ids else "[MISS]"
        rank = retrieved_ids.index(item["expected"]) + 1 if item["expected"] in retrieved_ids else "-"
        print(f'{hit} Query: "{item["query"]}"')
        print(f'   Expected: {item["expected"]} | Rank: {rank} | Retrieved: {retrieved_ids}')
        print()


if __name__ == "__main__":
    main()