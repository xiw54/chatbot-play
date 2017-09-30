import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
import io
from utils import *

# print(tf.__version__)

class Conversations(object):

    def __init__(self, file_path1='/Users/xinwang/DS502/chatbot_play/data/movie_lines.txt', file_path2='/Users/xinwang/DS502/chatbot_play/data/movie_conversations.txt'):
        # load data
        self.lines = io.open(file_path1, encoding='utf-8', errors='ignore').read().split('\n')
        self.conv_lines = io.open(file_path2, encoding='utf-8', errors='ignore').read().split('\n')

        self.questions = []
        self.answers = []
        self.clean_questions = []
        self.clean_answers = []
        self.lengths = []

        self.short_questions = None
        self.short_answers = None

        self.questions_vocab_to_int = None
        self.answers_vocab_to_int = None

        self.questions_int_to_vocab = None
        self.answers_int_to_vocab = None

        self.questions_int = None
        self.answers_int = None

        self.sorted_questions = None
        self.sorted_answers = None

    def process(self):
        id2line = {}
        for line in self.lines:
            _line = line.split(' +++$+++ ')
            if len(_line) == 5:
                id2line[_line[0]] = _line[4]

        convs = [ ]
        for line in self.conv_lines[:-1]:
            _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
            convs.append(_line.split(','))


        for conv in convs:
            for i in range(len(conv)-1):
                self.questions.append(id2line[conv[i]])
                self.answers.append(id2line[conv[i+1]])

        for question in self.questions:
            self.clean_questions.append(clean_text(question))

        for answer in self.answers:
            self.clean_answers.append(clean_text(answer))

        # Find the length of sentences
        for question in self.clean_questions:
            self.lengths.append(len(question.split()))
        for answer in self.clean_answers:
            self.lengths.append(len(answer.split()))

        # Remove questions and answers that are shorter than 2 words and longer than 20 words.
        self.short_questions, self.short_answers = remove_short_long_sent(self.clean_questions, self.clean_answers)

        # Create a dictionary for the frequency of the vocabulary
        vocab = create_vocab(self.short_questions, self.short_answers)

        # Remove rare words from the vocabulary.
        self.questions_vocab_to_int, self.answers_vocab_to_int = vocab_to_int(vocab)

        # Add the unique tokens to the vocabulary dictionaries.
        self.questions_vocab_to_int = add_code(self.questions_vocab_to_int)
        self.answers_vocab_to_int = add_code(self.answers_vocab_to_int)

        # Create dictionaries to map the unique integers to their respective words.
        self.questions_int_to_vocab = {v_i: v for v, v_i in self.questions_vocab_to_int.items()}
        self.answers_int_to_vocab = {v_i: v for v, v_i in self.answers_vocab_to_int.items()}


        # Add the end of sentence token to the end of every answer.
        for i in range(len(self.short_answers)):
            self.short_answers[i] += ' <EOS>'

        # Convert the text to integers.
        self.questions_int, self.answers_int = tex_to_int(self.short_questions, self.questions_vocab_to_int, self.short_answers, self.answers_vocab_to_int)


        # Sort questions and answers by the length of questions.
        self.sorted_questions, self.sorted_answers = sort_by_len(self.questions_int, self.answers_int)