def recall_at_k(retriever, test_set, k=3):
    hit = 0

    for item in test_set:
        query = item["query"]
        expected = item["expected"]

        results = retriever.search(query, top_k=k)

        retrieved_ids = [r["clause_id"] for r in results]

        if expected in retrieved_ids:
            hit += 1

    return hit / len(test_set)