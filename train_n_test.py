import tensorflow as tf
from lstm import *
from data_util import Conversations
import numpy as np
import time
from utils import clean_text

print(tf.__version__)
max_line_length = 20
# Set the Hyperparameters
epochs = 1000
batch_size = 128
rnn_size = 512
num_layers = 2
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.00001
learning_rate_decay = 0.9
min_learning_rate = 0.000001
keep_probability = 0.75
flags_train = True

# load dataset
conv = Conversations()
conv.process()

with tf.device('/gpu:0'):

    # Reset the graph to ensure that it is ready for training
    tf.reset_default_graph()
    # Start the session
    # sess = tf.InteractiveSession()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))

    # Create the training and inference logits

    # Load the model inputs
    input_data, targets, lr, keep_prob = model_inputs()
    # Sequence length will be the max line length for each batch
    sequence_length = tf.placeholder_with_default(max_line_length, None, name='sequence_length')
    # Find the shape of the input data for sequence_loss
    input_shape = tf.shape(input_data)

    train_logits, inference_logits = seq2seq_model(
        tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(conv.answers_vocab_to_int),
        len(conv.questions_vocab_to_int), encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers,
        conv.questions_vocab_to_int)

    # Create a tensor for the inference logits, needed if loading a checkpoint version of the model
    tf.identity(inference_logits, 'logits')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            train_logits,
            targets,
            tf.ones([input_shape[0], sequence_length]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    def pad_sentence_batch(sentence_batch, vocab_to_int):
        """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def batch_data(questions, answers, batch_size):
        """Batch questions and answers together"""
        for batch_i in range(0, len(questions)//batch_size):
            start_i = batch_i * batch_size
            questions_batch = questions[start_i:start_i + batch_size]
            answers_batch = answers[start_i:start_i + batch_size]
            pad_questions_batch = np.array(pad_sentence_batch(questions_batch, conv.questions_vocab_to_int))
            pad_answers_batch = np.array(pad_sentence_batch(answers_batch, conv.answers_vocab_to_int))
            yield pad_questions_batch, pad_answers_batch

    # Validate the training with 10% of the data
    train_valid_split = int(len(conv.sorted_questions)*0.15)

    # Split the questions and answers into training and validating data
    train_questions = conv.sorted_questions[train_valid_split:]
    train_answers = conv.sorted_answers[train_valid_split:]

    valid_questions = conv.sorted_questions[:train_valid_split]
    valid_answers = conv.sorted_answers[:train_valid_split]

    # print(len(train_questions))
    # print(len(valid_questions))

    display_step = 100  # Check training loss after every 100 batches
    stop_early = 0
    stop = 5  # If the validation loss does decrease in 5 consecutive checks, stop training
    validation_check = ((len(train_questions)) // batch_size // 2) - 1  # Modulus for checking validation loss
    total_train_loss = 0  # Record the training loss for each display step
    summary_valid_loss = []  # Record the validation loss for saving improvements in the model

    checkpoint = "model_lr_000001_cpu_fine_tune.ckpt"
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    if flags_train:
        saver.restore(sess, "//Users/xinwang/DS502/chatbot_play/models/model_lr_00001.ckpt")

        for epoch_i in range(1, epochs + 1):
            for batch_i, (questions_batch, answers_batch) in enumerate(
                    batch_data(train_questions, train_answers, batch_size)):
                start_time = time.time()
                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: questions_batch,
                     targets: answers_batch,
                     lr: learning_rate,
                     sequence_length: answers_batch.shape[1],
                     keep_prob: keep_probability})

                total_train_loss += loss
                end_time = time.time()
                batch_time = end_time - start_time

                if batch_i % display_step == 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  len(train_questions) // batch_size,
                                  total_train_loss / display_step,
                                  batch_time * display_step))
                    total_train_loss = 0

                if batch_i % validation_check == 0 and batch_i > 0:
                    total_valid_loss = 0
                    start_time = time.time()
                    for batch_ii, (questions_batch, answers_batch) in \
                            enumerate(batch_data(valid_questions, valid_answers, batch_size)):
                        valid_loss = sess.run(
                            cost, {input_data: questions_batch,
                                   targets: answers_batch,
                                   lr: learning_rate,
                                   sequence_length: answers_batch.shape[1],
                                   keep_prob: 1})
                        total_valid_loss += valid_loss
                    end_time = time.time()
                    batch_time = end_time - start_time
                    avg_valid_loss = total_valid_loss / (len(valid_questions) / batch_size)
                    print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time))

                    # Reduce learning rate, but not below its minimum value
                    learning_rate *= learning_rate_decay
                    if learning_rate < min_learning_rate:
                        learning_rate = min_learning_rate

                    summary_valid_loss.append(avg_valid_loss)
                    if avg_valid_loss <= min(summary_valid_loss):
                        print('New Record!')
                        stop_early = 0
                        saver.save(sess, checkpoint)

                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break

            if stop_early == stop:
                print("Stopping Training.")
                break

    else:

        saver.restore(sess, "//Users/xinwang/DS502/chatbot_play/models/model_lr_gpu_fine_tune_00001.ckpt")


        def question_to_seq(question, vocab_to_int):
            '''Prepare the question for the model'''

            question = clean_text(question)
            return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in question.split()]


        input_question = 'do you enjoy this course'

        # use a question from training set
        # Use a question from the data as your input
        # random = np.random.choice(len(conv.short_questions))
        # input_question = conv.short_questions[random]

        # Prepare the question
        input_question = question_to_seq(input_question, conv.questions_vocab_to_int)

        # Pad the questions until it equals the max_line_length
        input_question = input_question + [conv.questions_vocab_to_int["<PAD>"]] * (max_line_length - len(input_question))
        # Add empty questions so the the input_data is the correct shape
        batch_shell = np.zeros((batch_size, max_line_length))
        # Set the first question to be out input question
        batch_shell[0] = input_question

        answer_logits = sess.run(inference_logits, {input_data: batch_shell,
                                                    keep_prob: 1.0})[0]

        # Remove the padding from the Question and Answer
        pad_q = conv.questions_vocab_to_int["<PAD>"]
        pad_a = conv.answers_vocab_to_int["<PAD>"]

        print('Question:')
        # print('  Word Ids:      {}'.format([i for i in input_question if i != pad_q]))
        # print('  Input Words: {}'.format([conv.questions_int_to_vocab[i] for i in input_question if i != pad_q]))
        print(' '.join([conv.questions_int_to_vocab[i] for i in input_question if i != pad_q]))

        print('\nAnswer:')
        # print('  Word Ids:      {}'.format([i for i in np.argmax(answer_logits, 1) if i != pad_a]))
        # print(
        # '  Response Words: {}'.format([conv.answers_int_to_vocab[i] for i in np.argmax(answer_logits, 1) if i != pad_a]))
        print(' '.join([conv.answers_int_to_vocab[i] for i in np.argmax(answer_logits, 1) if i != pad_a]))
