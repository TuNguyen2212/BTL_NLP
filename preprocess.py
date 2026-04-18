from src.utils import read_file, write_lines, write_chunks
from src.clause_splitter import split_sentences, split_clauses
from src.np_chunker import np_chunk
from src.dependency_parser import parse_dependency
import json


def main():
    # 1. Read input
    text = read_file("input/raw_contracts.txt")

    # 2. Clause splitting
    sentences = split_sentences(text)

    all_clauses = []
    for s in sentences:
        clauses = split_clauses(s)
        all_clauses.extend(clauses)

    write_lines("output/clauses.txt", all_clauses)

    # 3. NP chunking
    all_chunks = []
    for clause in all_clauses:
        chunks = np_chunk(clause)
        all_chunks.append(chunks)

    write_chunks("output/chunks.txt", all_chunks)

    # 4. Dependency parsing
    all_dependencies = []
    for clause in all_clauses:
        deps = parse_dependency(clause)
        all_dependencies.append({
            "clause": clause,
            "dependencies": deps
        })

    with open("output/dependency.json", "w", encoding="utf-8") as f:
        json.dump(all_dependencies, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()