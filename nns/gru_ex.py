import os
import re

import corpus
import fire
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


ALL_CHARS = list(corpus.ALL_CHARS.values()) + [corpus.UNK] + [corpus.EOS]

CHR2IX = {v: k for k, v in enumerate(ALL_CHARS)}
IX2CHR = np.array(ALL_CHARS)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def read_dataset(path="data/corpus/combined.txt"):
    txt = open("data/corpus/combined.txt").read()
    unk_ix = CHR2IX[corpus.UNK]
    int_txt = np.array([CHR2IX.get(x, unk_ix) for x in txt])
    seq_length = 256
    examples_per_epoch = len(txt)//(seq_length+1)
    examples_per_epoch, len(txt)
    char_dataset = tf.data.Dataset.from_tensor_slices(int_txt)

    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    return dataset


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,
                                  embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True)


# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
def shuffle_dataset(dataset, batch_size=256, buffer_size=10000):
    shuffled_dataset = dataset.shuffle(
        buffer_size
    ).batch(batch_size, drop_remainder=True)

    return shuffled_dataset


def fit(model, dataset, epochs=1000, checkpoint_dir="./gru_ex_checkpoints"):
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    model.compile(optimizer='adam', loss=loss)

    shuffled_dataset = shuffle_dataset(dataset)

    history = model.fit(
        shuffled_dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    return history



def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

    # Converting our start string to numbers (vectorizing)
    input_eval = [CHR2IX[s] for s in start_string + corpus.EOS]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        new_char = IX2CHR[predicted_id]
        if new_char == corpus.EOS:
            return ''.join(text_generated)
        else:
            text_generated.append(new_char)

    return (start_string + ''.join(text_generated))


def learn(path="data/corpus/combined.txt",
          checkpoint_dir="./gru_ex_checkpoints"):
    ds = read_dataset(path)
    model = build_model(
        vocab_size=len(ALL_CHARS),
        embedding_dim=256,
        rnn_units=1024,
        batch_size=256,
    )
    model.summary()
    hist = fit(model, ds, checkpoint_dir=checkpoint_dir)


def load_model(checkpoint_dir="./gru_ex_checkpoints"):
    restored_model = build_model(len(ALL_CHARS),
                                 embedding_dim=256,
                                 rnn_units=1024,
                                 batch_size=1)
    restored_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    restored_model.build(tf.TensorShape([1, None]))

    restored_model.summary()

    return restored_model


def generate(text, checkpoint_dir="./gru_ex_checkpoints"):
    tf.get_logger().setLevel('ERROR')
    model = load_model(checkpoint_dir)
    print(generate_text(model, text))


if __name__ == "__main__":

    fire.Fire({
        'learn': learn,
        'generate': generate,
    })
