def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.strip() + "\n")


def write_chunks(path, all_chunks):
    with open(path, "w", encoding="utf-8") as f:
        for chunks in all_chunks:
            for word, tag in chunks:
                f.write(f"{word}\t{tag}\n")
            f.write("\n")