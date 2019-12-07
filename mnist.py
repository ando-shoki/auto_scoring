import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
#from tensorflow.keras.initializers import GlorotUniform
import tensorflow as tf
import numpy as np
import make_read_qr

classes = ["0","1","2","3","4","5","6","7","8","9"]
num_classes = len(classes)
image_size = 28

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#model = load_model('auto_scoring/model.h5')#学習済みモデルをロードする
model = tf.keras.models.load_model('model.h5')#学習済みモデルをロードする


@app.route('/',methods = ['GET','POST'])
def upload_file():
    #bugが出たらkerasがやばいかも
    #global graph
    #with graph.as_default():
    print("hoge")
    if request.method == 'POST':
        print('POSTNOW')
        '''
        @qr_file : 解答欄の元のQR画像ファイル
        @ans_file: 生徒が記入済みのQR画像ファイル
        '''
        if 'qr_file'or'ans_file' not in request.files:
            flash('ファイルがありません')#flashはユーザーに挙動が正しいかを知らせる
            return redirect(request.url)
        qr_file = request.files['qr_file']
        ans_file = request.files['ans_file']
        if qr_file.filename == '' or ans_file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if qr_file and allowed_file(qr_file.filename):
            if ans_file and allowd_file(ans_file.filename):
                print('qr and ans OK')
                qr_filename = secure_filename(qr_file.filename)
                ans_filename = secure_filename(ans_file.filename)
                return(qr_filename)
                print(ans_filename)
                #なぜかpathが存在しないって怒られた
                #file.save(filename)
                #filepath = filename
                qr_file.save(os.path.join(UPLOAD_FOLDER,qr_filename))
                ans_file.save(os.path.join(UPLOAD_FOLDER,ans_filename))
                qr_filepath = os.path.join(UPLOAD_FOLDER, qr_filename)
                ans_filepath = os.path.join(UPLOAD_FOLDER, ans_filename)

                #受け取った画像をnumpy形式に変換
                qr_img = image.load_img(qr_filepath ,grayscale=True,target_size=(image_size,image_size))
                qr_img = image.img_to_array(qr_img)
                ans_img = image.load_img(ans_filepath ,grayscale=True,target_size=(image_size,image_size))
                ans_img = image.img_to_array(ans_img)
                qr_data = np.array([qr_img])
                ans_data = np.array([ans_img])
                data = 255 - (ans_data - qr_data)
                result = model.predict(data)[0]
                predicted = result.argmax()
                pred_ans ='この画像は' + classes[predicted]+ 'です'

            return render_template('index.html', answer = pred_ans)
        else:
            return render_template('index.html', answer = '')
    return render_template('index.html', answer = '')
    
#直接実行した時のみに動く
if __name__ == '__main__':
    # qr_num = make_read_qr.read_qr(1)
    # print(qr_num.values.astype('i') == 1)
    # app.debug = True
    app.secret_key = 'super secret key'
    app.run(port=8000, debug=False)
        