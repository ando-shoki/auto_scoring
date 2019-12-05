import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
#from tensorflow.keras.initializers import GlorotUniform
import tensorflow as tf
import numpy as np

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
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('ファイルがありません')#flashはユーザーに挙動が正しいかを知らせる
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('ファイルがありません')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                #なぜかpathが存在しないって怒られた
                #file.save(filename)
                #filepath = filename
                file.save(os.path.join(UPLOAD_FOLDER,filename))
                filepath = os.path.join(UPLOAD_FOLDER, filename)

                #受け取った画像をnumpy形式に変換
                img = image.load_img(filepath ,grayscale=True,target_size=(image_size,image_size))
                img = image.img_to_array(img)
                data = np.array([img])
                data = 255 - data
                result = model.predict(data)[0]
                predicted = result.argmax()
                pred_ans ='この画像は' + classes[predicted]+ 'です'

            return render_template('index.html', answer = pred_ans)
        return render_template('index.html', answer = '')
    
#直接実行した時のみに動く
if __name__ == '__main__':
    app.run()
        