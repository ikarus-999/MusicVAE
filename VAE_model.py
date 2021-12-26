import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.Models import Model

class Encoder_sub(Model):
    def __init__(self, hidden_size=2048):
        super(Encoder, self).__init__()
        self.lstm = Bidirectional(LSTM(hidden_size))
        self.linear = Dense()
        self.linear_2 = Dense(self.latent_dim)
        self.norm = LayerNormalization(scale=False)
        self.final_size = hidden_size
    
    def encode(self, x):
        # x : input (batch, seq_len, n_feat)
        # z : latent (batch, latent_dim)
        # mu (batch, latent_dim)
        # std (batch, latent_dim)

        x, h, c = self.lstm(x)

        mu = self.norm(self.linear(h))
        std = tf.nn.softplus(self.linear_2(h))
        h = h.transpose(0, 1).reshape(-1, self.final_size)
        
        z = self.reparam(mu, std)
        return z, mu, std

    def reparam(self, mu, std):
        eps = tf.random.normal(shape=std.shape)
        return mu + (eps * std)

    def call(self, x):
        z, mu, std = self.encode(x)
        return z, mu, std

class CategoricalLstmDecoder(Model):
    def lstm_cell(self, hidden_size):
        x = LSTM(hidden_size)
        return x

    def __init__(self, output_size, n_lstm, hidden_size):
        super(CategoricalLstmDecoder, self).__init__()
        self.n_lstm = n_lstm
        self.lstm_cells = [self.lstm_cell(hidden_size) for _ in range(n_lstm)]
        self.output_size = output_size
        self.linear = Dense()
        self.linear_2 = Dense(output_size)
        
    def call(self, x, h, c, temp=1.):

        x, (h, c) = self.lstm_cells(x, (h, c))
        logits = self.linear_2(x) / temp
        prob = tf.nn.softmax(axis=2)(logits)
        out = tf.argmax(prob, 2)
        return out, prob, h, c
    
class MusicVAEmodel(Model):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(MusicVAEmodel, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.encoder = Encoder_sub()
        self.decoder = CategoricalLstmDecoder()
