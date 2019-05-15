import sys

sys.path.append('../') #root path
from data_utils import *
from models.ops import *

class Model_Train():
    def __init__(self, config):
        self.config = config
        self.build_model()
        log_dir = os.path.join(config.summary_dir)
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)


    def build_model(self):
        """ model """
        if self.config.exp_type == 0:
            self.generators = S_Net(num_metrics=self.config.num_metrics, structure_type= 'advanced')
        elif self.config.exp_type == 1:
            self.generators = S_Net_contskip(num_metrics=self.config.num_metrics, structure_type='advanced')
        elif self.config.exp_type == 2:
            self.generator = S_Net_progressiveskip(num_metrics=self.config.num_metrics, structure_type='advanced')
        elif self.config.exp_type == 3:
            self.generator = S_Net_intermediated_awared(num_metrics=self.config.num_metrics, structure_type='advanced')


        self.generator_optimizer = tf.keras.optimizers.Adam(self.config.learning_rate)

        """ saver """
        self.step = tf.Variable(0,dtype=tf.int64)
        self.ckpt = tf.train.Checkpoint(step=self.step,
                                        generator_optimizer=self.generator_optimizer,
                                        generator0=self.generators[0],
                                        #generator1=self.generators[1],
                                        #generator2=self.generators[2],
                                        #generator3=self.generators[3],
                                        #generator4=self.generators[4],
                                        #generator5=self.generators[5],
                                        #generator6=self.generators[6],
                                        #generator7=self.generators[7],
                                        )
        self.save_manager = tf.train.CheckpointManager(self.ckpt, self.config.checkpoint_dir, max_to_keep=3)
        self.save  = lambda : self.save_manager.save(checkpoint_number=self.step) #exaple : model.save()



    @tf.function
    def training(self, inputs):
        paired_input, paired_target = inputs
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            B_from_As = [g(paired_input) for g in self.generators]

            """ loss for generator """
            gen_losses = [L1loss(paired_target, B_from_A) for B_from_A in B_from_As]
            gen_loss = tf.reduce_mean(gen_losses)

        """ optimize """
        G_vars = []
        for g in self.generators:
            G_vars += g.trainable_variables
        G_vars = list(set(G_vars))
        generator_gradients = gen_tape.gradient(gen_loss, G_vars)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, G_vars))

        inputs_concat = tf.concat([paired_input, paired_target], axis=2)
        return_dicts = {"inputs_concat" :inputs_concat}
        return_dicts.update({'gen_loss{}'.format(e) : l  for e,l in enumerate(gen_losses)})
        return_dicts.update({'gen_loss' : gen_loss})
        return_dicts.update({'B_from_A{}'.format(e) : tf.concat([paired_input,l,paired_target],axis=2)  for e,l in enumerate(B_from_As)})
        return return_dicts



    def train_step(self,iterator, summary_name = "train", log_interval = 100):
        """ training """
        result_logs_dict = self.training(iterator.__next__())

        """ log summary """
        if summary_name and self.step.numpy() % log_interval == 0:
            with self.train_summary_writer.as_default():
                for key, value in result_logs_dict.items():
                    value = value.numpy()
                    if len(value.shape) == 0:
                        tf.summary.scalar("{}_{}".format(summary_name,key), value, step=self.step)
                    elif len(value.shape) in [3,4]:
                        tf.summary.image("{}_{}".format(summary_name, key), denormalize(value), step=self.step)


        """ return log str """
        return "g_loss : {}".format(result_logs_dict["gen_loss"])




    # Typically, the test dataset is not large
    """
    @tf.function
    def inference(self, model, input_image):
        # result = [g(input_image) for g in self.generators]
        for g in self.generators:
            result = g(input_image)
        return result
    """
    @tf.function
    def testing(self, test_dataset):
        outputs = [[] for _ in range(self.config.num_metrics)]
        losses = [[] for _ in range(self.config.num_metrics)]
        PSNRs = [[] for _ in range(self.config.num_metrics)]

        for input_image_test, label_image_test in test_dataset:
            B_from_As = [g(input_image_test) for g in self.generators]
            tf.print("inference")
            for e, B_from_A in enumerate(B_from_As):
                losses[e].append(L1loss(label_image_test, B_from_A))
                #outputs[e].append(tf.concat([input_image_test, B_from_A.numpy(), label_image_test], axis=2))
                # label_image_test_crop = edge_crop(label_image_test), B_from_A = edge_crop(B_from_A)
                # label_image_test_crop = cvt_ycbcr(label_image_test)[...,-1], B_from_A = cvt_ycbcr(B_from_A)[...,-1]
                PSNRs[e].append(tf.image.psnr(label_image_test, B_from_A, 1))
        return outputs,losses,PSNRs

    def test_step(self, test_dataset, summary_name = "test"):
        outputs, losses, PSNRs = self.testing(test_dataset.__iter__())
        '''
        """ log summary """
        if summary_name and self.step.numpy() %100 == 0:
            with self.train_summary_writer.as_default():
                for e, output in enumerate(outputs):
                    tf.summary.image("{}_B_from_A_{}_0".format(summary_name,e), denormalize(output[0]), step=self.step)
                    tf.summary.image("{}_B_from_A_{}_1".format(summary_name,e), denormalize(output[1]), step=self.step)
                for e, loss in enumerate(losses):
                    tf.summary.scalar("{}_loss_{}".format(summary_name, e),np.mean(loss), step=self.step)
                for e, PSNR in enumerate(PSNRs):
                    tf.summary.scalar("{}_psnr_{}".format(summary_name, e),np.mean(PSNR), step=self.step)

        """ return log str """
        log = "\n"
        for i in range(self.config.num_metrics):
            log += "[output{}] loss = {}, psnr = {}\n".format(i,np.mean(losses[i]),np.mean(PSNRs[i]))
        '''
        log = ""
        return log





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=4)  # -1 for CPU
    parser.add_argument("--crop_size", type=list, default=[512, 512], nargs="+", help='Image size after crop.')
    parser.add_argument("--buffer_size", type=int, default=20000, help='Data buffer size.')
    parser.add_argument("--batch_size", type=int, default=16, help='Minibatch size(global)')
    parser.add_argument("--patch_size", type=int, default=48, help='Minibatch size(global)')
    parser.add_argument("--jpeg_quality", type=int, default=20, help='Minibatch size(global)')
    parser.add_argument("--data_root_train", type=str, default='../dataset/train/BSD400', help='Data root dir')
    parser.add_argument("--data_root_test", type=str, default='../dataset/test/Set5', help='Data root dir')
    parser.add_argument("--channels", type=int, default=3, help='Channel size')
    parser.add_argument("--model_tag", type=str, default="default", help='Exp name to save logs/checkpoints.')
    parser.add_argument("--checkpoint_dir", type=str, default='../__outputs/checkpoints/', help='Dir for checkpoints.')
    parser.add_argument("--summary_dir", type=str, default='../__outputs/summaries/', help='Dir for tensorboard logs.')
    parser.add_argument("--restore_file", type=str, default=None, help='file for resotration')
    parser.add_argument("--graph_mode", type=bool, default=False, help='use graph mode for training')
    config = parser.parse_args()

    model  = Model_Train(config)



