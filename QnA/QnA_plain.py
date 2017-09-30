from data_util import Conversations
from utils import clean_text
import numpy as np
import tensorflow as tf

def question_to_seq(question, vocab_to_int):
    '''Prepare the question for the model'''

    question = clean_text(question)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in question.split()]

max_line_length = 20
batch_size = 128

# Create your own input question
input_question = 'How are you?'

# Use a question from the data as your input
# random = np.random.choice(len(short_questions))
# input_question = short_questions[random]

# Run the model with the input question
with tf.Session() as sess:
    tf.contrib
    # load graph
    # import_meta_graph appends the network defined in .meta file to the current graph.
    # this will create the graph/network for you but we still need to load the value of the parameters that we had trained on this graph.
    saver = tf.train.import_meta_graph('plain_model.ckpt.meta')
    saver.restore(sess, "/plain_model.ckpt.data-00000-of-00001")
    print("Model restored.")



    pred_ = sess.run(decoder_logits, feed_dict={
                         encoder_inputs: batch_,
                         decoder_inputs: din_,
                     })

    decoder_prediction = tf.argmax(decoder_logits, 2)

conv = Conversations()
conv.process()

# Prepare the question
input_question = question_to_seq(input_question, conv.questions_vocab_to_int)

# Pad the questions until it equals the max_line_length
input_question = input_question + [conv.questions_vocab_to_int["<PAD>"]] * (max_line_length - len(input_question))
# Add empty questions so the the input_data is the correct shape
batch_shell = np.zeros((batch_size, max_line_length))
# Set the first question to be out input question
batch_shell[0] = input_question

# Remove the padding from the Question and Answer
pad_q = conv.questions_vocab_to_int["<PAD>"]
pad_a = conv.answers_vocab_to_int["<PAD>"]

print('Question')
print('  Word Ids:      {}'.format([i for i in input_question if i != pad_q]))
print('  Input Words: {}'.format([conv.questions_int_to_vocab[i] for i in input_question if i != pad_q]))

print('\nAnswer')
print('  Word Ids:      {}'.format([i for i in np.argmax(answer_logits, 1) if i != pad_a]))
print('  Response Words: {}'.format([conv.answers_int_to_vocab[i] for i in np.argmax(answer_logits, 1) if i != pad_a]))