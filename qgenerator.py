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
    def __init__(self, qubits):
        self.alpha = np.random.uniform(low=0.0, high=np.pi)
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
    def __init__(self, qubits, n_layers):
        super().__init__(qubits)
        self.n_layers = n_layers

        for l in range(self.n_layers):
            prefix = 'layer'+str(l)
            self._layer_body(prefix)




class OneSubGANGenerator():
    def __init__(self,
                 n_layers,
                 n_qubits,
                 batch_size = 32,
                 generator_learning_rate=1e-2,
                 discriminator_learning_rate=1e-3,
                 num_measurements = None):

        self.BATCH_SIZE = batch_size

        self.n_layers =n_layers
        self.n_qubits = n_qubits
        self.qubits = [cirq.GridQubit(0,j) for j in range(self.n_qubits)]
        if num_measurements == None or num_measurements>self.n_qubits:
            self.measurement = [cirq.Z(self.qubits[j]) for j in range(self.n_qubits)]
        else:
            self.measurement = [cirq.Z(self.qubits[j]) for j in range(num_measurements)]

        data = datasets.load_digits(n_class=1)
        # (self.train_images, self.train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        (self.train_images, self.train_labels) = (np.array(data.data), np.array(data.target))

        # self.train_images = self.train_images.reshape(self.train_images.shape[0], 28, 28, 1).astype('float32')
        self.train_images = self.train_images.astype('float32')
        # self.train_images = (self.train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

        # Batch and shuffle the data
        self.NUM_SAMPLES = self.train_images.shape[0]
        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_images).shuffle(self.train_images.shape[0]).batch(self.BATCH_SIZE)


        self.img_dims = self.train_images.shape[1]

        self.q_data_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='circuits_input')
        self.qgenerator_circuit = QGANSubGenerator(self.qubits, self.n_layers).circuit
        differentiator = tfq.differentiators.ParameterShift()
        self.expectation_layer = tfq.layers.PQC(self.qgenerator_circuit,
                                                operators=self.measurement,
                                                repetitions=5000,
                                                differentiator=differentiator)

        self.generator = self.make_generator_model()
        self.discriminator = self.discriminator_network(self.img_dims,30)

        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_optimizer = tf.keras.optimizers.Adam(generator_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(discriminator_learning_rate)


    def make_generator_model(self):

        model = tf.keras.Sequential([
            self.q_data_input,
            self.expectation_layer,
            tf.keras.layers.Dense(self.img_dims, activation="relu")
        ])

        return model



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



    def noisy_quantum_input(self, num_samples=None):
        return tfq.convert_to_tensor([QInputGenerator(self.qubits).make_circuit() for _ in range(self.BATCH_SIZE if num_samples==None else num_samples)])



    def generate_and_save_images(self, epoch):
        noise = self.noisy_quantum_input(16)
        predictions = self.generator(noise, training=False)

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

        noise_circuits = self.noisy_quantum_input()

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise_circuits, training=True)

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

