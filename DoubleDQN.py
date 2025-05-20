import tensorflow as tf

class Qnetwork(tf.keras.Model):
    def __init__(self, s_size, a_size):
        super(Qnetwork, self).__init__()
        self.a_size = a_size
        self.s_size = s_size
        self.dense1 = tf.keras.layers.Dense(10, activation='relu', kernel_initializer='glorot_normal', name='w1')
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.dense2 = tf.keras.layers.Dense(6, activation='relu', kernel_initializer='glorot_normal', name='w2')
        self.dropout2 = tf.keras.layers.Dropout(0.25)
        self.out = tf.keras.layers.Dense(a_size, name='w3')

    def call(self, x, training=False):
        # Normalize input to [-1,1]
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = x / tf.constant([180.0, 180.0])
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        q_values = self.out(x)
        return q_values

    def predict_action(self, x):
        q_values = self.call(x, training=False)
        return tf.argmax(q_values, axis=1)

    def get_q_value(self, x, a):
        q_values = self.call(x, training=False)
        indices = tf.stack([tf.range(tf.shape(a)[0]), a], axis=1)
        return tf.gather_nd(q_values, indices)
