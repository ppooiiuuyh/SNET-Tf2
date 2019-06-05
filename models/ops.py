import tensorflow as tf
import numpy as np

""" =================================================================
 ops 
================================================================= """
def L1loss(input,target):
    #return tf.reduce_sum(tf.reduce_mean(tf.abs(input - target),axis=0))
    return tf.reduce_mean(tf.abs(input - target))




""" =================================================================
 model blocks
================================================================= """
initializer = tf.initializers.VarianceScaling()
def EncoderBlock(x, activation = tf.keras.layers.LeakyReLU(alpha=0.2), nf = 256):
    x = tf.keras.layers.Conv2D(nf, 5, strides=1, padding='same', kernel_initializer=initializer, use_bias=True)(x)
    x = activation(x)
    x = tf.keras.layers.Conv2D(nf, 5, strides=1, padding='same', kernel_initializer=initializer, use_bias=True)(x)
    x = activation(x)
    return x

def DecoderBlock(x, activation = tf.keras.layers.LeakyReLU(alpha=0.2), nf = 256):
    x = tf.keras.layers.Conv2D(nf, 5, strides=1, padding='same', kernel_initializer=initializer, use_bias=True)(x)
    x = activation(x)
    x = tf.keras.layers.Conv2D(3, 5, strides=1, padding='same', kernel_initializer=initializer, use_bias=True)(x)
    x = activation(x)
    return x



def ConvolutionalUnit(x, structure_type = 'classic', activation = tf.keras.layers.LeakyReLU(alpha=0.2), nf = 256):
    residual = x

    if structure_type == "classic":
        x = tf.keras.layers.Conv2D(nf, 5, strides=1, padding='same', kernel_initializer=initializer, use_bias=True)(x)
        x = activation(x)
        x = tf.keras.layers.Add()([x, residual])

    elif structure_type == "advanced":
        x = tf.keras.layers.Conv2D(nf, 5, strides=1, padding='same', kernel_initializer=initializer, use_bias=True)(x)
        x = activation(x)
        x = tf.keras.layers.Conv2D(nf, 5, strides=1, padding='same', kernel_initializer=initializer, use_bias=True)(x)
        x = tf.keras.layers.Lambda(lambda x: x * 0.1)(x)
        x = tf.keras.layers.Add()([x, residual])

    return x

def S_Net(channels = 3, num_metrics=3 , structure_type='classic', nf = 256):
    inputs = tf.keras.layers.Input(shape=[None, None, channels])
    encoder = EncoderBlock(inputs, nf = nf)
    convolution_units = []
    decoders = []
    for i in range(num_metrics):
        convolution_units.append(ConvolutionalUnit( convolution_units[-1] if len(convolution_units)>0 else ConvolutionalUnit(encoder, nf=nf), structure_type = structure_type, nf=nf))
        decoders.append(DecoderBlock(convolution_units[-1],nf=nf))
    #decoders = [DecoderBlock(cu) for cu in convolution_units]
    #return [tf.keras.Model(inputs=[inputs], outputs=[dec]) for dec in decoders]
    return tf.keras.Model(inputs=[inputs], outputs=decoders)



def S_Net_contskip(channels = 3, num_metrics=3 , structure_type='classic', nf= 256):
    inputs = tf.keras.layers.Input(shape=[None, None, channels])
    encoder = EncoderBlock(inputs,nf = nf)
    convolution_units = []
    decoders = []
    for i in range(num_metrics):
        convolution_units.append(ConvolutionalUnit( convolution_units[-1] if len(convolution_units)>0 else ConvolutionalUnit(encoder, nf= nf), structure_type = structure_type, nf= nf))
        decoders.append(tf.keras.layers.Add()([DecoderBlock(convolution_units[-1], nf=nf), inputs]))
    #decoders = [ tf.keras.layers.Add()([DecoderBlock(cu), inputs]) for cu in convolution_units]
    #return [tf.keras.Model(inputs=[inputs], outputs=[dec]) for dec in decoders]
    return tf.keras.Model(inputs=[inputs], outputs=decoders)



def S_Net_nonshared(channels = 3, num_metrics=3 , structure_type='classic', nf = 256):
    inputs = tf.keras.layers.Input(shape=[None, None, channels])
    encoder = EncoderBlock(inputs, nf = nf)
    convolution_units = []
    decoders = []
    for i in range(num_metrics):
        convolution_units.append(ConvolutionalUnit( convolution_units[-1] if len(convolution_units)>0 else ConvolutionalUnit(encoder, nf=nf), structure_type = structure_type, nf=nf))
        decoders.append(DecoderBlock(ConvolutionalUnit(convolution_units[-1],nf=nf),nf=nf))
    #decoders = [DecoderBlock(cu) for cu in convolution_units]
    #return [tf.keras.Model(inputs=[inputs], outputs=[dec]) for dec in decoders]
    return tf.keras.Model(inputs=[inputs], outputs=decoders)


def S_Net_contskip_nonshared(channels = 3, num_metrics=3 , structure_type='classic', nf = 256):
    inputs = tf.keras.layers.Input(shape=[None, None, channels])
    encoder = EncoderBlock(inputs, nf = nf)
    convolution_units = []
    decoders = []
    for i in range(num_metrics):
        convolution_units.append(ConvolutionalUnit( convolution_units[-1] if len(convolution_units)>0 else ConvolutionalUnit(encoder, nf=nf), structure_type = structure_type, nf=nf))
        decoders.append(tf.keras.layers.Add()(DecoderBlock(ConvolutionalUnit(convolution_units[-1],nf=nf),nf=nf), inputs))
    return tf.keras.Model(inputs=[inputs], outputs=decoders)



def S_Net_progressiveskip(channels = 3, num_metrics=3 , structure_type='classic'):
    inputs = tf.keras.layers.Input(shape=[None, None, channels])
    encoder = EncoderBlock(inputs)
    convolution_units = [ConvolutionalUnit(encoder)]
    for _ in range(1, num_metrics):
        convolution_units.append(ConvolutionalUnit( convolution_units[-1], structure_type = structure_type))

    decoders = []
    for e,cu in enumerate(convolution_units):
        decoders.append(tf.keras.layers.Add()([DecoderBlock(cu), inputs if e == 0 else decoders[-1]]))

    #return = [tf.keras.Model(inputs=[inputs], outputs=[dec]) for dec in decoders]
    return tf.keras.Model(inputs=[inputs], outputs=decoders)







def EncoderBlock_intermediate_awared(x):
    x = tf.keras.layers.Conv2D(256, 5, strides=1, padding='same', kernel_initializer=initializer, use_bias=True)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(256, 5, strides=1, padding='same', kernel_initializer=initializer, use_bias=True)(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def DecoderBlock_intermediate_awared(x,b):
    x = tf.keras.layers.concatenate([x,b],axis = -1)
    x = tf.keras.layers.Conv2D(256, 5, strides=1, padding='same', kernel_initializer=initializer, use_bias=True)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(3, 5, strides=1, padding='same', kernel_initializer=initializer, use_bias=True)(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def S_Net_intermediated_awared(channels = 3, num_metrics=3 , structure_type='classic'):
    inputs = tf.keras.layers.Input(shape=[None, None, channels])
    encoder = EncoderBlock_intermediate_awared(inputs)
    convolution_units = [ConvolutionalUnit(encoder)]
    for _ in range(1, num_metrics):
        convolution_units.append(ConvolutionalUnit( convolution_units[-1], structure_type = structure_type))

    decoders = []
    for e,cu in enumerate(convolution_units):
        decoders.append(DecoderBlock_intermediate_awared(cu, inputs if e == 0 else decoders[-1]))
    return tf.keras.Model(inputs=[inputs], outputs=decoders)


def S_Net_modular_firstblock(channels = 3, structure_type='classic'):
    inputs = tf.keras.layers.Input(shape=[None, None, channels])
    encoder = EncoderBlock(inputs)
    convolution_unit = ConvolutionalUnit(encoder)
    decoders = DecoderBlock(convolution_unit)
    return tf.keras.Model(inputs=[inputs], outputs=[decoders,convolution_unit])

def S_Net_modular_appendedblock(channels = 3, structure_type='classic'):
    inputs = tf.keras.layers.Input(shape=[None, None, 256])
    convolution_unit = ConvolutionalUnit(inputs)
    decoders = DecoderBlock(convolution_unit)
    return tf.keras.Model(inputs=[inputs], outputs=[decoders,convolution_unit])





if __name__ == "__main__":
    inputs = tf.keras.layers.Input(shape=[None, None, 3])

    x = np.random.random([1,512,512,3])
    x = EncoderBlock(x)
    print(x.shape)

    x = np.random.random([1,512,512,256])
    x = ConvolutionalUnit(x)
    print(x.shape)

    x = np.random.random([1,512,512,3]).astype(np.float32)
    outputs = S_Net(structure_type = "classic")([x])
    print(outputs)