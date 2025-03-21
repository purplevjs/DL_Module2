import jieba

def load_stopwords(file_path):
    stopwords = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.append(word)
    return stopwords

def simplify_text(text):
    # Word segmentation
    words = list(jieba.cut(text))

    stopwords = load_stopwords("stopwords.txt")
    # Remove stopwords
    filtered_words = [word for word in words if word not in stopwords]

    # Remove consecutive repeated words
    final_words = []
    for i, word in enumerate(filtered_words):
        if i == 0 or word != filtered_words[i - 1]:  # Only keep non-consecutively repeated words
            final_words.append(word)
    
    token_count = len(final_words)
    # Recombine into a string
    cleaned_text = "".join(final_words)

    return cleaned_text, token_count