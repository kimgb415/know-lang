from know_lang_bot.code_parser.parser import CodeParser

# Usage example:
if __name__ == "__main__":
    parser = CodeParser(".")
    chunks = parser.parse_repository()
    for chunk in chunks:
        print(f"{chunk.type}: {chunk.name} ({chunk.start_line}-{chunk.end_line})")