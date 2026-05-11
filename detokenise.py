import re
import black

def detokenize(snippet: str) -> str:
    """
    Convert custom tokenized code into readable Python code.
    """

    # Remove sequence markers
    snippet = snippet.replace("<s>", "")
    snippet = snippet.replace("</s>", "")

    tokens = snippet.split()

    indent = 0
    lines = []
    current_line = []

    for tok in tokens:

        # Handle newline
        if tok == "<EOL>":
            line = " ".join(current_line)

            # Fix spacing around punctuation/operators
            line = cleanup_spacing(line)

            lines.append("    " * indent + line)
            current_line = []
            continue

        # Handle indentation
        elif tok == "<INDENT>":
            indent += 1
            continue

        elif tok == "<DEDENT>":
            indent = max(indent - 1, 0)
            continue

        # Replace string literals
        elif tok.startswith("<STR_LIT"):
            current_line.append('"STR"')
            continue

        # Replace numeric literals
        elif tok.startswith("<NUM_LIT"):
            num = extract_num(tok)
            current_line.append(num)
            continue

        # Ignore start/end tokens
        elif tok in ("<s>", "</s>"):
            continue

        # Normal token
        current_line.append(tok)

    return "\n".join(lines)


def extract_num(tok: str) -> str:
    """
    Extract number from token like <NUM_LIT:10>
    """
    m = re.match(r"<NUM_LIT:(.*?)>", tok)
    if m:
        return m.group(1)

    return "0"


def cleanup_spacing(code: str) -> str:
    """
    Convert:
        foo ( x , y ) :
    into:
        foo(x, y):
    """

    # Remove spaces before punctuation
    code = re.sub(r"\s+([.,:;()\[\]{}])", r"\1", code)

    # Remove spaces after opening brackets
    code = re.sub(r"([(\[{])\s+", r"\1", code)

    # Fix operators
    operators = [
        r"\+", "-", r"\*", "/", "//", "%",
        "==", "!=", "<=", ">=", "<", ">",
        "=", r"\+=", "-=", r"\*=", "/="
    ]

    for op in operators:
        code = re.sub(rf"\s*{op}\s*", lambda m: f" {m.group(0).strip()} ", code)

    # Clean extra spaces
    code = re.sub(r"\s+", " ", code)

    return code.strip()




if __name__ == "__main__":

    sample = """
    <s> def add ( a , b ) : <EOL>
    <INDENT>
    return a + b <EOL>
    <DEDENT>
    </s>
    """

    result = detokenize(sample)
    formatted = black.format_str(result, mode=black.FileMode())
    print(formatted)
