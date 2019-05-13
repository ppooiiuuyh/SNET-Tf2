import datetime
import time

from backups.ops import *
from data_utils import *
from models.model_SNET import Model_Train

#tf.config.gpu.set_per_process_memory_fraction(0.6)
tf.config.gpu.set_per_process_memory_growth(True)

""" --------------------------------------------------------------------
configuaration
---------------------------------------------------------------------"""
start = time.time()
time_now = datetime.datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument("--exp_type", type=int, default=1, help='experiment type')
parser.add_argument("--gpu", type=str, default=4)  # -1 for CPU
parser.add_argument("--crop_size", type=list, default=[512, 512], nargs="+", help='Image size after crop.')
parser.add_argument("--buffer_size", type=int, default=20000, help='Data buffer size.')
parser.add_argument("--batch_size", type=int, default=16, help='Minibatch size(global)')
parser.add_argument("--patch_size", type=int, default=48, help='Minibatch size(global)')
parser.add_argument("--jpeg_quality", type=int, default=20, help='Minibatch size(global)')
parser.add_argument("--num_metrics", type=int, default=3, help='the number of metrics')
parser.add_argument("--data_root_train", type=str, default='./dataset/train/BSD400', help='Data root dir')
parser.add_argument("--data_root_test", type=str, default='./dataset/test/Set5', help='Data root dir')
parser.add_argument("--channels", type=int, default=3, help='Channel size')
parser.add_argument("--model_tag", type=str, default="default", help='Exp name to save logs/checkpoints.')
parser.add_argument("--checkpoint_dir", type=str, default='./__outputs/checkpoints/', help='Dir for checkpoints.')
parser.add_argument("--summary_dir", type=str, default='./__outputs/summaries/', help='Dir for tensorboard logs.')
parser.add_argument("--restore_file", type=str, default=None, help='file for resotration')
parser.add_argument("--graph_mode", type=bool, default=False, help='use graph mode for training')
config = parser.parse_args()


def generate_expname_automatically():
    name = "SNET_%s_%02d_%02d_%02d_%02d_%02d" % (config.model_tag,
            time_now.month, time_now.day, time_now.hour,
            time_now.minute, time_now.second)
    return name
expname  = generate_expname_automatically()
config.checkpoint_dir += expname ; check_folder(config.checkpoint_dir)
config.summary_dir += expname ; check_folder(config.summary_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)




""" --------------------------------------------------------------------
build model
---------------------------------------------------------------------"""
""" build model """
model = Model_Train(config)

""" restore model """
if config.restore_file is not None :
    model.ckpt.restore(config.restore_file)




""" --------------------------------------------------------------------
prepare dataset
---------------------------------------------------------------------"""
""" prepare paired iterator """
# train_iterator.__next__() = (paired_input, paired_target), unpaired_input, unpaired_target #repeat
# test_iterator.__iter__().__next__() = test_input #not repeat
# reference_iterator.__next__() = reference # repeat
#train_iterator, test_dataset, reference_iterator = make_iterator(config, line_normalizer= line_normalizer) #preaload all images on memory
train_iterator, test_dataset = make_iterator_ontime(config) #load each batches ontime





""" --------------------------------------------------------------------
train
---------------------------------------------------------------------"""
while True : #manuallry stopping
    """ train """
    log = model.train_step(train_iterator, log_interval= 100)
    print("[train] step:{} elapse:{} {}".format(model.step.numpy(), time.time() - start, log))


    """ test """
    if model.step.numpy() % 100 == 0:
        log = model.test_step(test_dataset, summary_name="test")
        print("[test] step:{} elapse:{} {}".format(model.step.numpy(), time.time() - start, log))


    """ save model """
    if model.step.numpy() % 100 == 0:  save_path = model.save()

    model.step.assign_add(1)
