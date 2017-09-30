import tensorflow as tf
import numpy as np
import utils
from lstm import process_encoding_input
from data_util import Conversations

# encoder starts with empty state and runs through the input sequence
# only final state is passed to decoder
# decoder uses encoder's final state as its initial state
# input for decoder are batch-sized matrix
# Decoder's outputs are mapped onto the output space using projection layer
# projection layer: [hidden_units X output_vocab_size]

# encoder and decoder RNNs expect dense vector representation of words, [batch_size, max_time, input_embedding_size]

def model_inputs():

    encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
    decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
    decoder_inputs = tf.placeholder(tf.int32, [None, None], name='decoder_inputs')

    return encoder_inputs, decoder_targets, decoder_inputs

def plain_seq2seq(answers_vocab_size, questions_vocab_size, enc_embedding_size, dec_embedding_size,
                  encoder_inputs, decoder_inputs, encoder_hidden_units, decoder_hidden_units):

    enc_embeddings = tf.Variable(tf.random_uniform([answers_vocab_size + 1, enc_embedding_size], 0, 1))
    dec_embeddings = tf.Variable(tf.random_uniform([questions_vocab_size + 1, dec_embedding_size], 0, 1))
    # given word 4, we represent it as 4th column of embedding matrix
    encoder_inputs_embedded = tf.nn.embedding_lookup(enc_embeddings, encoder_inputs)
    decoder_inputs_embedded = tf.nn.embedding_lookup(dec_embeddings, decoder_inputs)

    ####################  Encoder ######################
    # In seq2seq without attention this is the only point where Encoder passes information to Decoder.
    # We hope that backpropagation through time (BPTT) algorithm will tune the model to pass
    # enough information throught the thought vector for correct sequence output decoding.
    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        encoder_cell, encoder_inputs_embedded, time_major=False,
        dtype=tf.float32
    )

    del encoder_outputs

    ####################  Decoder ######################
    # Since we pass encoder_final_state as initial_state to the decoder they should be compatible.
    # This means the same cell type (LSTMCell in our case),
    # the same amount of hidden_units and the same amount of layers (single layer).
    with tf.variable_scope("decoding") as decoding_scope:

        decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
            decoder_cell, decoder_inputs_embedded,

            initial_state=encoder_final_state,

            dtype=tf.float32, time_major=False, scope="plain_decoder"
        )

        # At this point decoder_cell output is a hidden_units sized vector at every timestep.
        # However, for training and prediction we need logits of size vocab_size.
        # Reasonable thing would be to put linear layer (fully-connected layer without activation function)
        # on top of LSTM output to get non-normalized logits.
        # This layer is called projection layer by convention.

        decoder_train_logits = tf.contrib.layers.linear(decoder_outputs, questions_vocab_size)

    return decoder_train_logits, decoder_train_logits



if __name__ == '__main__':

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    PAD = 0
    EOS = 1

    vocab_size = 10
    input_embedding_size = 20

    encoder_hidden_units = 20
    decoder_hidden_units = encoder_hidden_units

    encoder_inputs, decoder_targets, decoder_inputs = model_inputs()
    decoder_logits, _ = plain_seq2seq(vocab_size, vocab_size, input_embedding_size, input_embedding_size,
                  encoder_inputs, decoder_inputs, encoder_hidden_units, decoder_hidden_units)

    decoder_prediction = tf.argmax(decoder_logits, 2)

    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
        logits=decoder_logits,
    )

    loss = tf.reduce_mean(stepwise_cross_entropy)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    sess.run(tf.global_variables_initializer())

    ################# Test #####################

    batch_ = [[6], [3, 4], [9, 8, 7]]

    batch_, batch_length_ = utils.batch(batch_)
    print('batch_encoded:\n' + str(batch_))

    din_, dlen_ = utils.batch(np.ones(shape=(3, 1), dtype=np.int32),
                                max_sequence_length=4)
    print('decoder inputs:\n' + str(din_))

    pred_ = sess.run(decoder_prediction,
                     feed_dict={
                         encoder_inputs: batch_,
                         decoder_inputs: din_,
                     })
    print('decoder predictions:\n' + str(pred_))

    ################# Trian #####################

    batch_size = 100

    batches = utils.random_sequences(length_from=3, length_to=8,
                                       vocab_lower=2, vocab_upper=10,
                                       batch_size=batch_size)


    def next_feed():
        batch = next(batches)
        encoder_inputs_, _ = utils.batch(batch)
        decoder_targets_, _ = utils.batch(
            [(sequence) + [EOS] for sequence in batch]
        )
        decoder_inputs_, _ = utils.batch(
            [[EOS] + (sequence) for sequence in batch]
        )
        return {
            encoder_inputs: encoder_inputs_,
            decoder_inputs: decoder_inputs_,
            decoder_targets: decoder_targets_,
        }


    loss_track = []
    max_batches = 3001
    batches_in_epoch = 1000

    try:
        for batch in range(max_batches):
            fd = next_feed()
            _, l = sess.run([train_op, loss], fd)
            loss_track.append(l)

            if batch == 0 or batch % batches_in_epoch == 0:
                print('batch {}'.format(batch))
                print('  minibatch loss: {}'.format(sess.run(loss, fd)))
                predict_ = sess.run(decoder_prediction, fd)
                for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                    print('  sample {}:'.format(i + 1))
                    print('    input     > {}'.format(inp))
                    print('    predicted > {}'.format(pred))
                    if i >= 2:
                        break
                print()
    except KeyboardInterrupt:
        print('training interrupted')
