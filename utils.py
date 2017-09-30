import re


def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text

def remove_short_long_sent(questions, answers, min_line_length = 2, max_line_length = 20):

    # Filter out the questions that are too short/long
    short_questions_temp = []
    short_answers_temp = []

    i = 0
    for question in questions:
        if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
            short_questions_temp.append(question)
            short_answers_temp.append(answers[i])
        i += 1

    # Filter out the answers that are too short/long
    short_questions = []
    short_answers = []

    i = 0
    for answer in short_answers_temp:
        if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
            short_answers.append(answer)
            short_questions.append(short_questions_temp[i])
        i += 1
    return short_questions, short_answers

def create_vocab(short_questions, short_answers):
    vocab = {}
    for question in short_questions:
        for word in question.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    for answer in short_answers:
        for word in answer.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
    return vocab


def vocab_to_int(vocab, threshold=10):
    questions_vocab_to_int = {}

    word_num = 0
    for word, count in vocab.items():
        if count >= threshold:
            questions_vocab_to_int[word] = word_num
            word_num += 1

    answers_vocab_to_int = {}

    word_num = 0
    for word, count in vocab.items():
        if count >= threshold:
            answers_vocab_to_int[word] = word_num
            word_num += 1
    return questions_vocab_to_int, answers_vocab_to_int

def add_code(vocab_to_int):
    codes = ['<PAD>', '<EOS>', '<UNK>', '<GO>']

    for code in codes:
        vocab_to_int[code] = len(vocab_to_int) + 1
    return vocab_to_int

def tex_to_int(short_questions, questions_vocab_to_int, short_answers, answers_vocab_to_int):

    # Replace any words that are not in the respective vocabulary with <UNK>
    questions_int = []
    for question in short_questions:
        ints = []
        for word in question.split():
            if word not in questions_vocab_to_int:
                ints.append(questions_vocab_to_int['<UNK>'])
            else:
                ints.append(questions_vocab_to_int[word])
        questions_int.append(ints)

    answers_int = []
    for answer in short_answers:
        ints = []
        for word in answer.split():
            if word not in answers_vocab_to_int:
                ints.append(answers_vocab_to_int['<UNK>'])
            else:
                ints.append(answers_vocab_to_int[word])
        answers_int.append(ints)
    return questions_int, answers_int

def sort_by_len(questions_int, answers_int, max_line_length=20):
    sorted_questions = []
    sorted_answers = []

    for length in range(1, max_line_length + 1):
        for i in enumerate(questions_int):
            if len(i[1]) == length:
                sorted_questions.append(questions_int[i[0]])
                sorted_answers.append(answers_int[i[0]])

    return sorted_questions, sorted_answers