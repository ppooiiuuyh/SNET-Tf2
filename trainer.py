import datetime
import time
from data_utils2 import *
from models.model_SNET import Model_Train
#tf.config.gpu.set_per_process_memory_fraction(0.6)
#tf.config.gpu.set_per_process_memory_growth(True)

""" --------------------------------------------------------------------
configuaration
---------------------------------------------------------------------"""
start = time.time()
time_now = datetime.datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument("--exp_type", type=int, default=1, help='experiment type')
parser.add_argument("--gpu", type=str, default=0)  # -1 for CPU
parser.add_argument("--crop_size", type=list, default=[512, 512], nargs="+", help='Image size after crop.')
parser.add_argument("--buffer_size", type=int, default=20000, help='Data buffer size.')
parser.add_argument("--batch_size", type=int, default=16, help='Minibatch size(global)')
parser.add_argument("--patch_size", type=int, default=48, help='Minipatch size(global)')
parser.add_argument("--jpeg_quality", type=int, default=20, help='jpeg quallity')
parser.add_argument("--num_metrics", type=int, default=8, help='the number of metrics')
parser.add_argument("--num_filters", type=int, default=256, help='the number of filters')
parser.add_argument("--learning_rate", type=float, default=0.0001, help="lr")
parser.add_argument("--min_learning_rate", type=float, default=0.000001, help="min_lr")
parser.add_argument("--data_root_train", type=str, default="/projects/datasets/restoration/DIV2K/", help='Data root dir')
parser.add_argument("--data_root_test", type=str, default="/projects/datasets/restoration/LIVE1/", help='Data root dir')
#parser.add_argument("--data_root_test", type=str, default="./dataset/test/Set5", help='Data root dir')
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
trainset_dispenser = Trainset_Dispenser(data_path=config.data_root_train, config=config)
testset_dispenser = Testset_Dispenser(data_path=config.data_root_test, config=config)
# train_iterator.__next__() = (paired_input, paired_target), unpaired_input, unpaired_target #repeat
# test_iterator.__iter__().__next__() = test_input #not repeat
# reference_iterator.__next__() = reference # repeat





""" --------------------------------------------------------------------
train
---------------------------------------------------------------------"""
while True : #manuallry stopping
    """ train """
    log, output = model.train_step(trainset_dispenser, log_interval= 100)
    if model.step.numpy() % 1 == 0:
        print("[train] step:{} elapse:{} {}".format(model.step.numpy(), time.time() - start, log))

        #visualization
        output_concat = np.concatenate([output[i] for i in range(len(output))], axis=1)[0]
        output_concat = cv2.resize(output_concat,(output_concat.shape[1]*3,output_concat.shape[0]*3))
        cv2.imshow('image', output_concat[...,::-1])
        cv2.waitKey(10)

    """ test """
    if model.step.numpy() % 5000 == 0:
        log = model.test_step(testset_dispenser, summary_name="test")
        print("[test] step:{} elapse:{} {}".format(model.step.numpy(), time.time() - start, log))


    """ save model """
    '''
    if model.step.numpy() % 50000 == 0:  save_path = model.save()
    '''

    model.step.assign_add(1)
