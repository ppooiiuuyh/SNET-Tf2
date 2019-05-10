import sys
sys.path.append('../')
from models.generator import *
from data_utils import *


class Model_Tester():
    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):# Build Generator and Discriminator
        self.generatorAB = Augcycle_Generator(self.config.channels)

        self.step = tf.Variable(0,dtype=tf.int64)
        self.ckpt = tf.train.Checkpoint(step=self.step, generatorAB=self.generatorAB)

    @tf.function()
    def inference(self, x, style_image = None):
        ZB = tf.random.normal([1, 8])
        return self.generatorAB([x,ZB])

if __name__ == "__main__":
    """ --------------------------------------------------------------------
    configuaration
    ---------------------------------------------------------------------"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=4)  # -1 for CPU
    parser.add_argument("--image_size", type=list, default=[512, 512], nargs="+", help='Image size before crop.')
    parser.add_argument("--crop_size", type=list, default=[512, 512], nargs="+", help='Image size after crop.')
    parser.add_argument("--data_root", type=str, default='/datasets/line_stylizer_data/safebooru_lines_245', help='Data root dir')
    parser.add_argument("--channels", type=int, default=1, help='Channel size')
    parser.add_argument("--restore_file", type=str, default="./pretrained/ckpt-0", help='file for resotration')
    config = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

    model = Model_Tester(config)
    model.ckpt.restore(config.restore_file)
    print(model.step)

