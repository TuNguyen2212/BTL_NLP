import os
import json

from config import INPUT_PATH, OUTPUT_DIR, CLAUSES_PATH, CHUNKS_PATH, DEPENDENCY_PATH
from src.utils import read_file, write_lines, write_chunks
from src.contract_cleaner import clean_contracts
from src.clause_splitter import split_sentences, split_clauses
from src.np_chunker import np_chunk
from src.dependency_parser import parse_dependency


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_text = read_file(INPUT_PATH)
    text = clean_contracts(raw_text)
    print(f"[Cleaner] Extracted {len(text.splitlines())} legal paragraphs from input.")

    sentences = split_sentences(text)
    all_clauses = []
    for s in sentences:
        clauses = split_clauses(s)
        all_clauses.extend(clauses)
    write_lines(CLAUSES_PATH, all_clauses)

    all_chunks = []
    for clause in all_clauses:
        chunks = np_chunk(clause)
        all_chunks.append(chunks)
    write_chunks(CHUNKS_PATH, all_chunks)

    all_dependencies = []
    for clause in all_clauses:
        try:
            deps = parse_dependency(clause)
        except Exception as e:
            print(f"[Dep] WARNING: parse failed for clause: {clause[:60]!r} -> {e}")
            deps = []
        all_dependencies.append({"clause": clause, "dependencies": deps})

    with open(DEPENDENCY_PATH, "w", encoding="utf-8") as f:
        json.dump(all_dependencies, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
