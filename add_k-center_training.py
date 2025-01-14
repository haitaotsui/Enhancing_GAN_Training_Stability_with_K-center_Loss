import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Define the network structure of the generator and discriminator
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.hidden1 = Dense(128, activation='relu')
        self.output_layer = Dense(784, activation='tanh')

    def call(self, z):
        hidden = self.hidden1(z)
        output = self.output_layer(hidden)
        return output

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.hidden1 = Dense(128, activation='relu')
        self.logits = Dense(1)
        self.output_layer = tf.keras.layers.Activation('sigmoid')

    def call(self, X):
        hidden = self.hidden1(X)
        logits = self.logits(hidden)
        output = self.output_layer(logits)
        return output, logits

# Define the generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Define loss function and optimizer
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
D_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
G_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Define the training process
@tf.function
def train_step(real_images, z):
    # Train the discriminator
    with tf.GradientTape() as D_tape:
        fake_images = generator(z)
        real_output, real_logits = discriminator(real_images)
        fake_output, fake_logits = discriminator(fake_images)
        D_real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        D_fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        D_loss = D_real_loss + D_fake_loss
    D_gradients = D_tape.gradient(D_loss, discriminator.trainable_variables)
    D_optimizer.apply_gradients(zip(D_gradients, discriminator.trainable_variables))

    # Training the generator
    with tf.GradientTape() as G_tape:
        fake_images = generator(z)
        fake_output, _ = discriminator(fake_images)
        G_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    G_gradients = G_tape.gradient(G_loss, generator.trainable_variables)
    G_optimizer.apply_gradients(zip(G_gradients, generator.trainable_variables))

    return D_loss, G_loss
    
# Train the model
batch_size = 100
epochs = 100

# Assume the mnist dataset has been loaded
# Loading using TensorFlow's built-in dataset
(mnist_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
mnist_images = mnist_images.reshape(-1, 784).astype('float32') / 255.0 * 2 - 1  # Standardized to [-1, 1]
dataset = tf.data.Dataset.from_tensor_slices(mnist_images).shuffle(60000).batch(batch_size)

for epoch in range(epochs):
    for batch_images in dataset:
        z = np.random.uniform(-1, 1, size=(batch_size, 100)).astype('float32')
        D_loss, G_loss = train_step(batch_images, z)

    # Output the loss value at regular intervals
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Discriminator loss: {D_loss.numpy()}, Generator loss: {G_loss.numpy()}")

# Generate adversarial samples and save
sample_z = np.random.uniform(-1, 1, size=(batch_size, 100)).astype('float32')
generated_samples = generator(sample_z)
samples = (generated_samples.numpy()[:16] + 1) / 2  # Denormalize to [0, 1]
fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(7, 7))
for ax, img in zip(axes.flatten(), samples):
    ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
    ax.axis('off')
plt.savefig('generated_samples.png')
