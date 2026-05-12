import re


# Delimiters required by the assignment:
# whitespaces, tabs, digits, and characters:
# ()[]{}.!?,;:+=-_"'`~#@&*%€$§\/
TOKEN_SPLIT_REGEX = r"[\s\t\d\(\)\[\]\{\}\.\!\?\,\;\:\+\=\-\_\"\'\`\~\#\@\&\*\%\€\$\§\\\/]+"


def load_stopwords(stopwords_path):
    """
    Load stopwords from a text file.

    Each line in stopwords.txt is expected to contain one stopword.
    All stopwords are converted to lowercase.
    """
    with open(stopwords_path, "r", encoding="utf-8") as file:
        return set(line.strip().lower() for line in file if line.strip())


def tokenize_text(text, stopwords):
    """
    Tokenize and clean review text according to assignment requirements.

    Steps:
    1. Case folding
    2. Tokenization using required delimiters
    3. Stopword removal
    4. Removal of one-character tokens
    """
    if text is None:
        return []

    text = text.lower()
    raw_tokens = re.split(TOKEN_SPLIT_REGEX, text)

    tokens = [
        token
        for token in raw_tokens
        if token and len(token) > 1 and token not in stopwords
    ]

    return tokens