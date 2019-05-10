import sys
sys.path.append('../')
from ops import *
import numpy as np

def GeneratorUnet(channels):
    #initializer = tf.random_normal_initializer(0., 0.02)
    initializer = tf.initializers.VarianceScaling()

    def conv_block (x, filters, size, strides, initializer=initializer):
        x = tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',kernel_initializer=initializer, use_bias=True)(x)
        x = InstanceNorm()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        return x


    inputs = tf.keras.layers.Input(shape=[None, None, channels])

    #encoding
    x0 = conv_block(inputs,32,5,2)
    x0 = conv_block(x0,64,3,1)
    x1 = conv_block(x0, 128, 3, 2)
    x1 = conv_block(x1, 128, 3, 1)
    x = conv_block(x1, 256, 3, 2)
    x = conv_block(x, 256, 3, 1)


    #decoding
    x = Tfresize(2.0,2.0)(x)
    x = conv_block(x,256,3,1)
    x = tf.keras.layers.concatenate([x, x1], axis = -1)
    x = conv_block(x,256,3,1)
    x = conv_block(x,128,3,1)

    x = Tfresize(2.0,2.0)(x)
    x = conv_block(x,128,3,1)
    x = tf.keras.layers.concatenate([x, x0], axis = -1)
    x = conv_block(x,128,3,1)
    x = conv_block(x,64,3,1)

    x = Tfresize(2.0,2.0)(x)
    x = conv_block(x,64,3,1)
    x = conv_block(x,32,3,1)

    output = tf.keras.layers.Conv2D(channels, 3, strides=1, padding='same',kernel_initializer=initializer, use_bias=True)(x)
    output = InstanceNorm()(output)
    output = tf.keras.layers.Activation('sigmoid')(output)

    return tf.keras.Model(inputs=inputs, outputs=output)



def Augcycle_Generator(channels, nlatent = 8):
    initializer = tf.initializers.VarianceScaling()

    def conv_block (x, filters, size, strides, padding = "SAME", initializer=initializer, activation =tf.keras.layers.LeakyReLU(alpha=0.2)):
        x = SpecConv2DLayer( filters, size, strides, kernel_initializer= initializer, padding = padding)(x)
        #x = InstanceNorm()(x)
        x = activation(x)
        return x

    def residual_with_adaptive_insnorm(x, mu, sigma,  filters, size, strides, initializer=initializer):
        input = x

        x = SpecConv2DLayer( filters, size, strides)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = AdaptiveInstanceNorm()([x,mu,sigma])
        x = SpecConv2DLayer( filters, size, strides)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Add()([x, input])
        return x



    inputs = tf.keras.layers.Input(shape=[None, None, channels])
    noise = tf.keras.layers.Input(shape=[nlatent])

    """ downsample """
    x = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [4, 4], [4, 4], [0, 0]], "SYMMETRIC"))(inputs)
    x = conv_block(x, 32, 9 , 2, padding = 'VALID')
    x = conv_block(x, 64, 3 , 2)
    x = conv_block(x, 128, 3 , 2)
    x = conv_block(x, 256, 3 , 1)

    """ adaptive insnorm """
    z = tf.keras.layers.Dense(256, activation = "relu")(noise)
    z = tf.keras.layers.Dense(256, activation = "relu")(z)
    mu = tf.keras.layers.Dense(256)(z)
    mu = tf.keras.layers.Reshape([1,1,256])(mu)
    sigma = tf.keras.layers.Dense(256)(z)
    sigma = tf.keras.layers.Reshape([1,1,256])(sigma)

    x = residual_with_adaptive_insnorm(x, mu, sigma, 256, 3, 1)
    x = residual_with_adaptive_insnorm(x, mu, sigma, 256, 3, 1)

    """ upsample """
    x = Tfresize(2.0,2.0)(x)
    x = conv_block(x,128,3,1)
    x = Tfresize(2.0,2.0)(x)
    x = conv_block(x,128,3,1)
    x = Tfresize(2.0,2.0)(x)
    x = conv_block(x,64,3,1)

    x = conv_block(x,channels,3,1, activation=tf.keras.layers.Activation('sigmoid'))
    output = x

    return tf.keras.Model(inputs=[inputs,noise], outputs=output)




def Augcycle_Latentencoder(channels, nlatent = 8):
    initializer = tf.initializers.VarianceScaling()
    def conv_block (x, filters, size, strides, padding = 'SAME', initializer=initializer):
        x = tf.keras.layers.Conv2D(filters, size, strides=strides, padding=padding ,kernel_initializer=initializer, use_bias=True)(x)
        x = InstanceNorm()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        return x

    inputs_x = tf.keras.layers.Input(shape=[None, None, channels])
    inputs_y = tf.keras.layers.Input(shape=[None, None, channels])


    x = tf.keras.layers.concatenate([inputs_x, inputs_y], axis = -1)
    x = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [4, 4], [4, 4], [0, 0]], "SYMMETRIC"))(x)
    x = conv_block(x,32,9,2, padding = 'VALID')
    x = conv_block(x,64,3,2)
    x = conv_block(x,128,3,2)
    x = conv_block(x,256,3,1)
    x = Reduce_____(tf.reduce_mean,[1,2])(x)
    x = tf.keras.layers.Dense(nlatent)(x)
    output = x

    return tf.keras.Model(inputs=[inputs_x,inputs_y], outputs=output)





if __name__ == "__main__":
    dummy_input = np.random.random([4,512,512,1]).astype(np.float32)
    dummy_noise = np.random.random([4,8]).astype(np.float32)
    """
    generator = Augcycle_Latentencoder(1)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        output = generator([dummy_input,dummy_input])
        loss = L1loss(1, output)

    generator_gradients = gen_tape.gradient(loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients,  generator.trainable_variables))
    print(generator_gradients)

    """
    generator = Augcycle_Generator(1)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        output = generator([dummy_input,dummy_noise])
        #print(output)
        loss = L1loss(1, output)

    #generator_gradients = gen_tape.gradient(loss, generator.trainable_variables)
    #generator_optimizer.apply_gradients(zip(generator_gradients,  generator.trainable_variables))
