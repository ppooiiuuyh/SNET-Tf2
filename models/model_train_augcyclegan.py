import sys

sys.path.append('../') #root path
from models.generator import *
from models.discriminator import *
from data_utils import *


class Model_Train():
    def __init__(self, config):
        self.config = config
        self.build_model()
        log_dir = os.path.join(config.summary_dir)
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)

    def build_model(self):# Build Generator and Discriminator
        """ model """
        self.generatorAB = Augcycle_Generator(self.config.channels)
        self.latent_encoderZA = Augcycle_Latentencoder(self.config.channels)
        self.generatorBA = Augcycle_Generator(self.config.channels)
        self.latent_encoderZB = Augcycle_Latentencoder(self.config.channels)


        self.discriminator_latentZA = Augcycle_Discriminator_latent()
        self.discriminator_latentZB = Augcycle_Discriminator_latent()
        self.discriminatorA = Discriminator_Patch(self.config.channels)
        self.discriminatorB = Discriminator_Patch(self.config.channels)

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
        #self.generator.summary()

        """ saver """
        self.step = tf.Variable(0,dtype=tf.int64)
        self.ckpt = tf.train.Checkpoint(step=self.step,
                                    generator_optimizer=self.generator_optimizer,
                                    discriminator_optimizer=self.discriminator_optimizer,
                                    generatorAB=self.generatorAB,
                                    generatorBA=self.generatorBA,
                                    latent_encoderZA=self.latent_encoderZA,
                                    latent_encoderZB=self.latent_encoderZB,
                                    discriminator_latentZA=self.discriminator_latentZA,
                                    discriminator_latentZB=self.discriminator_latentZB,
                                    discriminatorA=self.discriminatorA,
                                    discriminatorB = self.discriminatorB)
        self.save_manager = tf.train.CheckpointManager(self.ckpt, self.config.checkpoint_dir, max_to_keep=3)
        self.save  = lambda : self.save_manager.save(checkpoint_number=self.step) #exaple : model.save()

    @tf.function
    def training(self, inputs):
        (paired_input, paired_target), unpaired_input, unpaired_target = inputs
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            ZA = tf.random.normal([self.config.batch_size,8])
            ZB = tf.random.normal([self.config.batch_size,8])

            B_from_A = self.generatorAB([unpaired_input,ZB])
            za_from_AB = self.latent_encoderZA([unpaired_input,B_from_A])
            A_from_B = self.generatorBA([unpaired_target,ZA])
            zb_from_BA = self.latent_encoderZB([unpaired_target,A_from_B])

            B_from_A_guided = self.generatorAB([unpaired_input, zb_from_BA])
            A_from_B_guided = self.generatorBA([unpaired_target, za_from_AB])

            recon_A = self.generatorBA([B_from_A, za_from_AB])
            recon_zb = self.latent_encoderZB([B_from_A, unpaired_input])
            recon_B = self.generatorAB([A_from_B, zb_from_BA])
            recon_za = self.latent_encoderZA([A_from_B, unpaired_target])

            #recon_A_with_Z_A = self.generatorBA([B_from_A, ZA])
            #recon_B_with_Z_B = self.generatorAB([A_from_B, ZB])


            B_from_pairedA = self.generatorAB([paired_input,ZA])
            A_from_pairedB = self.generatorBA([paired_target,ZB])

            """ loss for discriminator """
            '''
            #dicriminator A
            discA_real_output = self.discriminatorA(unpaired_input, training=True)
            discA_generated_output = self.discriminatorA(A_from_B, training=True)
            discA_loss = getHingeDLoss(discA_real_output, discA_generated_output)

            #dicriminator B
            discB_real_output = self.discriminatorB(unpaired_target, training=True)
            discB_generated_output = self.discriminatorB(B_from_A, training=True)
            discB_loss = getHingeDLoss(discB_real_output, discB_generated_output)

            #discriminator ZA
            discZA_real_output = self.discriminator_latentZA(ZA, training=True)
            discZA_generated_output = self.discriminator_latentZA(za_from_AB, training=True)
            discZA_loss = discriminator_adv_loss(discZA_real_output, discZA_generated_output)

            #dicriminator ZB
            discZB_real_output = self.discriminator_latentZB(ZB, training=True)
            discZB_generated_output = self.discriminator_latentZB(zb_from_BA, training=True)
            discZB_loss = discriminator_adv_loss(discZB_real_output, discZB_generated_output)
            '''
            # dicriminator A
            discA_real_outputs = discriminator_multiscale(unpaired_input, disc_fn = self.discriminatorA)
            discA_generated_outputs = discriminator_multiscale(A_from_B, disc_fn = self.discriminatorA)
            discA_loss = tf.reduce_sum([getHingeDLoss(discA_real_output, discA_generated_output) for discA_real_output,discA_generated_output in zip(discA_real_outputs,discA_generated_outputs) ])

            # dicriminator B
            discB_real_outputs = discriminator_multiscale(unpaired_target, disc_fn = self.discriminatorB)
            discB_generated_outputs = discriminator_multiscale(B_from_A, disc_fn = self.discriminatorB)
            discB_loss = tf.reduce_sum([getHingeDLoss(discB_real_output, discB_generated_output) for discB_real_output,discB_generated_output in zip(discB_real_outputs,discB_generated_outputs) ])

            #discriminator ZA
            discZA_real_output = self.discriminator_latentZA(ZA, training=True)
            discZA_generated_output = self.discriminator_latentZA(za_from_AB, training=True)
            discZA_loss = discriminator_adv_loss(discZA_real_output, discZA_generated_output)

            #dicriminator ZB
            discZB_real_output = self.discriminator_latentZB(ZB, training=True)
            discZB_generated_output = self.discriminator_latentZB(zb_from_BA, training=True)
            discZB_loss = discriminator_adv_loss(discZB_real_output, discZB_generated_output)

            disc_loss = discA_loss + discB_loss + discZA_loss + discZB_loss


            """ loss for generator """
            #adversarial loss
            gen_adv_lossA =  tf.reduce_sum([getHingeGLoss(disc_generated_output) for disc_generated_output in discA_generated_outputs])
            gen_adv_lossB =  tf.reduce_sum([getHingeGLoss(disc_generated_output) for disc_generated_output in discB_generated_outputs])
            gen_adv_lossZA = getHingeGLoss(discZA_generated_output)
            gen_adv_lossZB = getHingeGLoss(discZB_generated_output)

            #cycle consistency loss
            loss_cyc_ABA = L1loss(recon_A, unpaired_input) * 1
            loss_cyc_BAB = L1loss(recon_B, unpaired_target)  * 1
            loss_cyc_zb2zb = L1loss(recon_zb, ZB) * 0.025  # authors used 0.025
            loss_cyc_za2za = L1loss(recon_za, ZA) * 0.025  # authors used 0.025

            #supervised loss
            gen_sup_lossA = L1loss(paired_input, A_from_pairedB) * self.config.sup_loss
            gen_sup_lossB = L1loss(paired_target, B_from_pairedA) * self.config.sup_loss

            gen_cycle_loss = loss_cyc_ABA + loss_cyc_BAB + loss_cyc_zb2zb + loss_cyc_za2za
            gen_adv_loss = gen_adv_lossA + gen_adv_lossB + gen_adv_lossZA + gen_adv_lossZB
            gen_sup_loss = gen_sup_lossA + gen_sup_lossB
            gen_loss = gen_cycle_loss +  gen_adv_loss   + gen_sup_loss

            genA_loss = gen_adv_lossA + gen_sup_lossA
            genB_loss = gen_adv_lossB + gen_sup_lossB



        """ optimize """
        G_vars = self.generatorAB.trainable_variables + self.generatorBA.trainable_variables + self.latent_encoderZA.trainable_variables + self.latent_encoderZB.trainable_variables
        D_vars = self.discriminatorA.trainable_variables + self.discriminatorB.trainable_variables + self.discriminator_latentZA.trainable_variables + self.discriminator_latentZB.trainable_variables
        generator_gradients = gen_tape.gradient(gen_loss, G_vars)
        discriminator_gradients = disc_tape.gradient(disc_loss, D_vars)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, D_vars))
        self.generator_optimizer.apply_gradients(zip(generator_gradients, G_vars))

        inputs_concat = tf.concat([unpaired_input, unpaired_target, paired_input, paired_target], axis=2)
        return {"gen_loss":gen_loss,"disc_loss":disc_loss,
                "discA_loss": discA_loss, "discB_loss": discB_loss,
                "genA_loss": genA_loss, "genA_adv_loss": gen_adv_lossA, "genA_sup_loss" : gen_sup_lossA,
                "genB_loss": genB_loss, "genB_adv_loss": gen_adv_lossB, "genB_sup_loss": gen_sup_lossA,
                "gen_cycle_loss" : gen_cycle_loss, "gen_adv_loss": gen_adv_loss,  "gen_sup_loss" : gen_sup_loss,

                "B_from_A": B_from_A, "A_from_B": A_from_B, "B_from_A_guided": B_from_A_guided, "A_from_B_guided": A_from_B_guided,
                "inputs_concat" :inputs_concat}


    def train_step(self,iterator, summary_name = "train", log_interval = 100):
        """ training """
        results = self.training(iterator.__next__())


        """ log summary """
        if summary_name and self.step.numpy() % log_interval == 0:
            with self.train_summary_writer.as_default():
                tf.summary.image("{}_unpaired_input, unpaired_target, paired_input, paired_target,".format(summary_name), denormalize(results["inputs_concat"].numpy()), step=self.step)
                tf.summary.image("{}_B_from_A".format(summary_name), denormalize(results["B_from_A"].numpy()), step=self.step)
                tf.summary.image("{}_A_from_B".format(summary_name), denormalize(results["A_from_B"].numpy()), step=self.step)
                tf.summary.image("{}_B_from_A_guided".format(summary_name), denormalize(results["B_from_A_guided"].numpy()), step=self.step)
                tf.summary.image("{}_A_from_B_guided".format(summary_name), denormalize(results["A_from_B_guided"].numpy()), step=self.step)

                tf.summary.scalar("{}_gen_loss".format(summary_name), results["gen_loss"], step=self.step)
                tf.summary.scalar("{}_disc_loss".format(summary_name), results["disc_loss"], step=self.step)
                tf.summary.scalar("{}_discA_loss".format(summary_name), results["discA_loss"], step=self.step)
                tf.summary.scalar("{}_discB_loss".format(summary_name), results["discB_loss"], step=self.step)
                tf.summary.scalar("{}_genA_loss".format(summary_name), results["genA_loss"], step=self.step)
                tf.summary.scalar("{}_genA_adv_loss".format(summary_name), results["genA_adv_loss"], step=self.step)
                tf.summary.scalar("{}_genA_sup_loss".format(summary_name), results["genA_sup_loss"], step=self.step)
                tf.summary.scalar("{}_genB_loss".format(summary_name), results["genB_loss"], step=self.step)
                tf.summary.scalar("{}_genB_adv_loss".format(summary_name), results["genB_adv_loss"], step=self.step)
                tf.summary.scalar("{}_genB_sup_loss".format(summary_name), results["genB_sup_loss"], step=self.step)
                tf.summary.scalar("{}_gen_cycle_loss".format(summary_name), results["gen_cycle_loss"], step=self.step)
                tf.summary.scalar("{}_gen_adv_loss".format(summary_name), results["gen_adv_loss"], step=self.step)
                tf.summary.scalar("{}_gen_sup_loss".format(summary_name), results["gen_sup_loss"], step=self.step)

        """ return log str """
        return "g_loss : {} d_loss : {}".format(results["gen_loss"], results["disc_loss"])


    # Typically, the test dataset is not large
    def test_step(self, iterator, reference_iterator, summary_name = "test"):
        self.B_from_As_val = []
        self.A_from_Bs_val = []
        self.B_from_A_guideds_val = []
        self.A_from_B_guideds_val = []

        for input_image_test in iterator:
            reference = reference_iterator.__next__()
            ZA = tf.random.normal([self.config.batch_size,8])
            ZB = tf.random.normal([self.config.batch_size,8])

            B_from_A = self.generatorAB([input_image_test, ZB])
            za_from_AB = self.latent_encoderZA([input_image_test, B_from_A])

            A_from_B = self.generatorBA([reference, ZA])
            zb_from_BA = self.latent_encoderZA([reference, A_from_B])

            B_from_A_guided = self.generatorAB([input_image_test, zb_from_BA])
            A_from_B_guided = self.generatorBA([reference, za_from_AB])

            self.B_from_As_val.append(tf.concat([input_image_test,B_from_A],axis=2))
            self.A_from_Bs_val.append(tf.concat([reference,A_from_B],axis=2))
            self.B_from_A_guideds_val.append(tf.concat([input_image_test,B_from_A_guided,reference],axis=2))
            self.A_from_B_guideds_val.append(tf.concat([reference,A_from_B_guided,input_image_test],axis=2))

        """ log summary """
        if summary_name and self.step.numpy() %100 == 0:
            with self.train_summary_writer.as_default():
                tf.summary.image("{}_B_from_A".format(summary_name), denormalize(tf.concat(self.B_from_As_val, axis=0).numpy()), step=self.step)
                tf.summary.image("{}_A_from_B".format(summary_name), denormalize(tf.concat(self.A_from_Bs_val, axis=0).numpy()), step=self.step)
                tf.summary.image("{}_B_from_A_guided".format(summary_name), denormalize(tf.concat(self.B_from_A_guideds_val, axis=0).numpy()), step=self.step)
                tf.summary.image("{}_A_from_B_guided".format(summary_name), denormalize(tf.concat(self.A_from_B_guideds_val, axis=0).numpy()), step=self.step)

        """ return log str """
        return ""





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=4)  # -1 for CPU
    parser.add_argument("--crop_size", type=list, default=[512, 512], nargs="+", help='Image size after crop.')
    parser.add_argument("--buffer_size", type=int, default=20000, help='Data buffer size.')
    parser.add_argument("--batch_size", type=int, default=1, help='Minibatch size(global)')
    parser.add_argument("--data_root", type=str, default='/datasets/line_stylizer_data/', help='Data root dir')
    parser.add_argument("--channels", type=int, default=1, help='Channel size')
    parser.add_argument("--model_tag", type=str, default="default", help='Exp name to save logs/checkpoints.')
    parser.add_argument("--checkpoint_dir", type=str, default='../__outputs/checkpoints/', help='Dir for checkpoints.')
    parser.add_argument("--summary_dir", type=str, default='../__outputs/summaries/', help='Dir for tensorboard logs.')
    parser.add_argument("--restore_file", type=str, default=None, help='file for resotration')

    parser.add_argument("--sup_loss", type=float, default=0.01, help='weights for supervised loss')

    config = parser.parse_args()

    model  = Model_Train(config)



