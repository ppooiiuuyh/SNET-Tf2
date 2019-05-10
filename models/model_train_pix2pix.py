import sys

sys.path.append('../') #root path
from line_stylizer.models.generator import *
from line_stylizer.models.discriminator import *
from line_stylizer.data_utils import *
from functools import partial

class Model_Train():
    def __init__(self, config):
        self.config = config
        self.build_model()
        log_dir = os.path.join(config.summary_dir)
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)

    def build_model(self):# Build Generator and Discriminator
        """ model """
        self.generator = GeneratorUnet(self.config.channels)
        self.discriminator = Discriminator_Pair(self.config.channels)

        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        #self.generator.summary()

        self.step = tf.Variable(0,dtype=tf.int64)
        self.ckpt = tf.train.Checkpoint(step=self.step,
                                   generator_optimizer=self.generator_optimizer,
                                   discriminator_optimizer=self.discriminator_optimizer,
                                   generator=self.generator,
                                   discriminator=self.discriminator)

        """ saver """
        self.save_manager = tf.train.CheckpointManager(self.ckpt, self.config.checkpoint_dir, max_to_keep=3)
        self.save  = partial(self.save_manager.save,checkpoint_number = self.step.numpy()) #exaple : model.save()



    def train_step(self,inputs):
        (paired_input, paired_target), unpaired_input, unpaired_target = inputs
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            self.gen_output = self.generator(paired_input, training=True)
            disc_real_output = self.discriminator([paired_input, paired_target], training=True)
            disc_generated_output = self.discriminator([paired_input, self.gen_output], training=True)

            self.gen_loss = generator_loss(disc_generated_output, self.gen_output, paired_target)
            self.disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(self.gen_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(self.disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))


    # Typically, the test dataset is not large. Do eagerly
    def val_step(self,inputs, summary_name = "val"):
        (paired_input, paired_target), unpaired_input, unpaired_target = inputs
        self.gen_output = self.generator(paired_input, training=True)
        disc_real_output = self.discriminator([paired_input, paired_target], training=True)
        disc_generated_output = self.discriminator([paired_input, self.gen_output], training=True)

        self.gen_loss = generator_loss(disc_generated_output, self.gen_output, paired_target)
        self.disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        """ log summary """
        if summary_name and self.step.numpy() %100 == 0:
            with self.train_summary_writer.as_default():
                tf.summary.image("{}_image".format(summary_name), denormalize(tf.concat([paired_input,self.gen_output,paired_target], axis=2).numpy()), step=self.step)
                tf.summary.scalar("{}_g_loss".format(summary_name), self.gen_loss.numpy(), step=self.step)
                tf.summary.scalar("{}_d_loss".format(summary_name), self.disc_loss.numpy(), step=self.step)

        """ return log str """
        return "g_loss : {} d_loss : {}".format(self.gen_loss, self.disc_loss)




    # Typically, the test dataset is not large
    def test_step(self, iterator, summary_name = "test"):
        self.images_val = []
        for input_image_test in iterator:
            gen_output_test = self.generator(input_image_test, training=False)
            self.images_val.append(tf.concat([input_image_test,gen_output_test],axis=2))

        """ log summary """
        if summary_name and self.step.numpy() %100 == 0:
            with self.train_summary_writer.as_default():
                tf.summary.image("{}_image".format(summary_name), denormalize(tf.concat(self.images_val, axis=0).numpy()), step=self.step)

        """ return log str """
        return ""



