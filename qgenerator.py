import cirq
import sympy
from tqdm import tqdm
import tensorflow as tf
import tensorflow_quantum as tfq
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

import time

from IPython import display

class QInputGenerator():
    def __init__(self, qubits, alpha):
        # self.alpha = np.random.uniform(low=0.0, high=np.pi)
        self.alpha = alpha
        self.qu = qubits
        self.circuit = cirq.Circuit()

    def make_circuit(self):
        gate_op = cirq.ry(self.alpha)
        for i, _ in enumerate(self.qu):
            self.circuit.append([gate_op.on(self.qu[i])])

        return self.circuit



class QSubGeneratorBuilder():
    def __init__(self, qubits):
        self.qu = qubits
        self.theta_symbols = []
        self.circuit = cirq.Circuit()

    def _u_ent(self):
        """
        Quantum circuit for entangling qubits with CZ gates
        Returns:
            none
        """
        for i in range(len(self.qu)):
            self.circuit.append([cirq.Z.controlled().on(self.qu[i],self.qu[(i+1) % len(self.qu)])])

    def _layer_body(self, prefix):
        """
        Construct the main body of the quantum circuit of the generator
        Args:
        Returns:
            none
        """
        for i, q in enumerate(self.qu):
            symbol = sympy.Symbol(prefix + '_' + str(i))
            self.theta_symbols.append(symbol)
            gate_ry = cirq.ry(symbol)
            self.circuit.append([gate_ry.on(q)])
        self._u_ent()



class QGANSubGenerator(QSubGeneratorBuilder):
    def __init__(self, qubits, n_layers, index):
        super().__init__(qubits)
        self.n_layers = n_layers
        self.index = index

        for l in range(self.n_layers):
            prefix = str(index)+'layer'+str(l)
            self._layer_body(prefix)

# class PatchConcatLayer(tf.keras.layers.Layer):
#     def __init__(self, batch_size, n_measurements, n_sub_gen):
#         super(PatchConcatLayer, self).__init__()
#         self.batch_size = batch_size
#         self.n_measurements = n_measurements
#         self.n_sub_gen = n_sub_gen
#         self.repeats_list = [2**n_measurements for _ in range(n_sub_gen)]
#         self.scale_dic = {'0':1.0/2.0, '1':-1.0/2.0}
#         op_arr = []
#         for i in range(self.n_sub_gen):
#             for j in range(2**self.n_measurements):
#                 b_arr = '{0:{fill}{align}5}'.format('0', fill='0', align='<') + '{0:b}'.format(j)
#                 b_arr = b_arr[::-1][0:self.n_measurements]
#                 for b in b_arr:
#                     op_arr.append(self.scale_dic[b])
#         op_arr = [op_arr]
#         self.op_tensor = tf.convert_to_tensor(op_arr)
#         self.op_tensor = tf.repeat(self.op_tensor, repeats=[batch_size], axis=0)


#     def call(self, inputs):
#         inp_reshape = tf.reshape(inputs, shape=[-1, self.n_sub_gen, self.n_measurements])
#         inp_extended = tf.repeat(inp_reshape, repeats=self.repeats_list, axis=2)
#         inp_extended = tf.reshape(inp_extended, shape=[-1, self.n_sub_gen*(2**self.n_measurements)])

#         inp_scaled = inp_extended*self.op_tensor
#         out = inp_scaled + 1

#         return out

class PatchConcatLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits, n_layers, sub_gen_list, img_dims, symbols_list, use_mlp=True):
        super().__init__()
        self.sub_gen_list = sub_gen_list
        self.img_dims = img_dims
        self.symbols = symbols_list
        self.use_mlp = use_mlp
        self.mlp = tf.keras.layers.Dense(self.img_dims, activation="relu")

        # init_tensor = tf.random_uniform_initializer(shape=(len(sub_gen_list), n_qubits*n_layers))
        # theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        # self.thetas = tf.Variable(initial_value=theta_init(shape=(len(sub_gen_list), n_qubits*n_layers)), trainable=True)


    def call(self, inputs):
        out_gen_list = []
        for i, inp in enumerate(inputs):
            if self.use_mlp:
                # out_gen_list.append(self.sub_gen_list[i](inp, symbol_names=self.symbols[i], symbol_values=self.thetas[i,:]))
                out_gen_list.append(self.sub_gen_list[i](inp))
            else:
                out_gen_list.append(tf.math.abs(self.sub_gen_list[i](inp)))

        out_sub_gen = tf.keras.layers.concatenate(out_gen_list, axis=1)
        if self.use_mlp:
            return self.mlp(out_sub_gen)
        else:
            return out_sub_gen




class OneSubGANGenerator():
    def __init__(self,
                 n_layers,
                 n_qubits,
                 n_sub_gen=4,
                 batch_size = 32,
                 generator_learning_rate=1e-2,
                 discriminator_learning_rate=1e-3,
                 use_mlp=True,
                 classic_generator=False,
                 n_neurons = 1000,
                 num_measurements = None):

        self.classic_generator = classic_generator
        self.BATCH_SIZE = batch_size

        self.n_layers =n_layers
        self.n_qubits = n_qubits
        self.n_sub_gen = n_sub_gen

        data = datasets.load_digits(n_class=1)
        # (self.train_images, self.train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        (self.train_images, self.train_labels) = (np.array(data.data), np.array(data.target))

        # self.train_images = self.train_images.reshape(self.train_images.shape[0], 28, 28, 1).astype('float32')
        self.train_images = self.train_images.astype('float32')

        # self.n_measurements =  np.floor(np.log2(self.train_images.shape[1]//self.n_sub_gen)).astype(int)
        # self.qubits = [cirq.GridQubit(0,j) for j in range(self.n_qubits)]
        # self.measurement = [cirq.Z(self.qubits[j]) for j in range(self.n_measurements)]

        # if num_measurements == None or num_measurements>self.n_qubits:
        #     self.measurement = [cirq.Z(self.qubits[j]) for j in range(self.n_qubits - self.n_measurements)]
        # else:
        #     self.measurement = [cirq.Z(self.qubits[j]) for j in range(num_measurements)]
        # self.train_images = (self.train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

        # Batch and shuffle the data
        self.NUM_SAMPLES = self.train_images.shape[0]
        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_images).shuffle(self.train_images.shape[0]).batch(self.BATCH_SIZE)


        self.img_dims = self.train_images.shape[1]

        # self.q_data_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='circuits_input')
        # self.qgenerator_circuit = QGANSubGenerator(self.qubits, self.n_layers).circuit
        self.qubits_list = []
        # self.measurement_list = []
        self.sub_gen_circuit_list = []
        self.sub_gen_list = []
        self.symbols_list = []
        for i in range(self.n_sub_gen):
            qubits = [cirq.GridQubit(i,j) for j in range(self.n_qubits)]
            self.qubits_list.append(qubits)

            qgenerator_circuit = QGANSubGenerator(qubits, self.n_layers, i)
            self.sub_gen_circuit_list.append(qgenerator_circuit.circuit)
            self.symbols_list.append(qgenerator_circuit.theta_symbols)
            if use_mlp:
                measurement = [cirq.Z(qubits[j]) for j in range(self.n_qubits)]
            else:
                measurement = cirq.measure(*qubits)

            q_data_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='circuits_input')
            if use_mlp:
                sub_gen = tfq.layers.PQC(qgenerator_circuit.circuit,
                                        operators=measurement,
                                        repetitions=1000,
                                        differentiator=tfq.differentiators.ParameterShift())
            else:
                sub_gen = tfq.layers.State()
            # sub_gen = tfq.layers.State()
            qu_layer = tf.keras.Sequential([q_data_input, sub_gen])
            self.sub_gen_list.append(qu_layer)

        # self.expectation_layer = tf.keras.layers.concatenate(self.sub_gen_list, axis=1)
        # self.patch_concat_layer = PatchConcatLayer(self.BATCH_SIZE, self.n_measurements, self.n_sub_gen)
        # self.patch_concat_layer = PatchConcatLayer(self.sub_gen_list)

        # self.generator = self.make_generator_model()
        if self.classic_generator:
            self.generator = tf.keras.Sequential([
                                tf.keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
                                tf.keras.layers.Dense(n_neurons, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
                                tf.keras.layers.Dense(self.img_dims, activation=tf.keras.layers.LeakyReLU(alpha=0.01))
                            ])
        else:
            self.generator = PatchConcatLayer(n_qubits, n_layers, self.sub_gen_list, self.img_dims, self.symbols_list, use_mlp=use_mlp)
        self.discriminator = self.discriminator_network(self.img_dims,30)

        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_optimizer = tf.keras.optimizers.Adam(generator_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(discriminator_learning_rate)


    # def make_generator_model(self):

    #     model = tf.keras.Sequential([
    #         self.expectation_layer,
    #         self.patch_concat_layer,
    #         tf.keras.layers.Dense(self.img_dims, activation="relu")
    #     ])

    #     return model



    def discriminator_network(self, num_nodes_1, num_nodes_2, alpha_1=0.01, alpha_2=0.01):
        """
        Constructs a three-layer classical discriminator in a sequential model using keras
        Args:
            num_nodes_1 (int): # of nodes in the first layer
            num_nodes_2 (int): # of nodes in the second layer
            alpha_1 (float): the slope for values lower than the threshold in the activation function
            alpha_2 (float): the slope for values lower than the threshold in the activation function
        Returns:
            a
        """
        return tf.keras.Sequential([
            tf.keras.layers.Dense(num_nodes_1, activation=tf.keras.layers.LeakyReLU(alpha=alpha_1)),
            tf.keras.layers.Dense(num_nodes_2, activation=tf.keras.layers.LeakyReLU(alpha=alpha_2)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])




    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss



    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)



    def noisy_quantum_input(self, alpha, index, num_samples=None):
        return tfq.convert_to_tensor([QInputGenerator(self.qubits_list[index], alpha).make_circuit() for _ in range(self.BATCH_SIZE if num_samples==None else num_samples)])



    def generate_and_save_images(self, epoch):
        alpha = np.random.uniform(low=0.0, high=np.pi)
        noise_list = []
        if self.classic_generator:
            noise_list = tf.random.uniform(shape=(16,1),minval=0.0, maxval=np.pi)
        else:
            for i in range(self.n_sub_gen):
                noise_circuits = self.noisy_quantum_input(alpha, i, 16)
                noise_list.append(noise_circuits)


        predictions = self.generator(noise_list, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i,:].numpy().reshape(8,8) *255, cmap='gray')
            plt.axis('off')

        # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()


    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, images):
        alpha = np.random.uniform(low=0.0, high=np.pi)
        noise_list = []
        if self.classic_generator:
            noise_list = tf.random.uniform(shape=(self.BATCH_SIZE,1),minval=0.0, maxval=np.pi)
        else:
            for i in range(self.n_sub_gen):
                noise_circuits = self.noisy_quantum_input(alpha, i)
                noise_list.append(noise_circuits)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise_list, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                         self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                             self.discriminator.trainable_variables))



    def train(self, epochs, show_images=False):
        for epoch in range(epochs):

            with tqdm(total=(self.NUM_SAMPLES//self.BATCH_SIZE)) as t:
                for image_batch in self.train_dataset:
                    self.train_step(image_batch)
                    t.update()

            if show_images:
                display.clear_output(wait=True)
                self.generate_and_save_images(epoch + 1,)

            # Save the model every 15 epochs
            # if (epoch + 1) % 15 == 0:
                # checkpoint.save(file_prefix = checkpoint_prefix)

