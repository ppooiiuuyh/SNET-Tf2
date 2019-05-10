import sys

sys.path.append('./demo_web')
from models.model_inference import  Model_Tester as Line_Stylier_Model_Tester
from pretrained_line_normalizer.line_normalizer_model_inference import Model_Tester as Line_Normalizer_Model_Tester
import numpy as np
from data_utils import *
from flask import Flask, request, send_from_directory, render_template, make_response
import cv2, argparse
from werkzeug import secure_filename
import tensorflow as tf
#import scipy.misc


UPLOAD_FOLDER = './demo_web/uploads'
RESULT_FOLDER = './demo_web/results'
ALLOWED_EXTENTIONS = set(['jpg', 'png', 'jpeg', 'PNG', 'JPG', 'JPEG'])
check_folder(UPLOAD_FOLDER)
check_folder(RESULT_FOLDER)
tf.config.gpu.set_per_process_memory_fraction(0.6)
tf.config.gpu.set_per_process_memory_growth(True)

app = Flask(__name__, static_url_path='/demo_web/static', template_folder='demo_web/templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1] in ALLOWED_EXTENTIONS


@app.after_request
def add_header(response):
    '''
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    '''
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            # if allowed_file(file.filename)
            filename = secure_filename(file.filename)
            file.save(os.path.normcase(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
        else:
            filename = request.cookies.get('filename')

        """ parse options """
        print("parse options")
        option_scale = float(request.form.get('scale_controller'))/100

        """ preprocess data """
        print("preprocess data")
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename), cv2.IMREAD_GRAYSCALE)
        bigger_size = max(img.shape[0], img.shape[1])
        if bigger_size > 1024:
            mult = 1024 / float(bigger_size)
            new_s = (int(img.shape[1]*mult), int(img.shape[0]*mult))
            img = cv2.resize(img,new_s)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), img)


        new_w = int(img.shape[1]*option_scale)  - int(img.shape[1]*option_scale)%8
        new_h = int(img.shape[0]*option_scale)  - int(img.shape[0]*option_scale)%8
        img_resized = cv2.resize(img, (new_w,new_h))
        img_resized = img_resized.reshape(1,img_resized.shape[0],img_resized.shape[1],1)
        img_resized = (img_resized/255.0).astype(np.float32)


        """ run model """
        print("run model")
        img_line_norm_result = line_normalizer_model.inference(img_resized)
        img_line_sty_result = line_stylier_model.inference(img_line_norm_result).numpy()
        img_line_sty_result = img_line_sty_result.reshape(img_line_sty_result.shape[1],img_line_sty_result.shape[2])

        #img_line_norm_result = img_resized.reshape(img_resized.shape[1],img_resized.shape[2])
        output_s1 = cv2.resize(img_line_sty_result, (img.shape[1], img.shape[0]))
        output_s1 = (output_s1 * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], "res_s1_" + filename), output_s1)


        """ make response """
        print("make response")
        random_int = np.random.randint(0, 3123819)
        #return render_template('hello.html', filename=filename, random_int=random_int)
        resp = make_response(render_template('main.html', filename=filename, random_int=random_int, scale=option_scale*100))

        resp.set_cookie('filename', filename)
        return resp

    return render_template('init.html')



@app.route('/results/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'],
            filename)

@app.route('/uploads/<filename>')
def uploads_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
            filename)


if __name__ == '__main__':
    """ --------------------------------------------------------------------
    configuaration
    ---------------------------------------------------------------------"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=4)  # -1 for CPU
    parser.add_argument("--image_size", type=list, default=[512, 512], nargs="+", help='Image size before crop.')
    parser.add_argument("--crop_size", type=list, default=[512, 512], nargs="+", help='Image size after crop.')
    parser.add_argument("--channels", type=int, default=1, help='Channel size')
    parser.add_argument("--restore_file", type=str, default="./pretrained/ckpt-0", help='file for resotration')
    parser.add_argument("--port", type=int, default=8888, help='port')

    config = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    #trategy = tf.distribute.MirroredStrategy()
    #with strategy.scope() :
    line_normalizer_model = Line_Normalizer_Model_Tester(config.channels)
    line_stylier_model = Line_Stylier_Model_Tester(config)
    line_stylier_model.ckpt.restore(config.restore_file)

    app.run(host='0.0.0.0', port=config.port)
    #app.run(host='0.0.0.0', port=8888)
