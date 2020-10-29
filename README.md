# Face-Recognition
Face Recognition using Python's OpenCV and TensorFlow

## Recommended Python Version: Python 3.7

## Getting Started

### Python Packages to be installed:
* numpy: <strong>pip install numpy</strong>
* matplotlib: <strong>pip install matplotlib</strong>
* openCV: <strong>pip install opencv-python</strong>
* keras: <strong>pip install keras</strong>

### Haar Cascade Classifier to detect the Face
In order to detect a frontal face from an image or video stream, we would require a haar cascade classifier. 
For this purpose we'll be using the file: <strong>haarcascade-frontalface-default.xml</strong>

### Collect your images to train your model to detect your face
1. Open the File: <b>train-test-set-generator.py</b>
2. In line 4: Provide the path of the HaarCascade "Classifier XML File:-<br>
<b>line 4: face_classifier = cv2.CascadeClassifier('PATH/haarcascade-frontalface-default.xml')</b>
3. In line 34: Provide the path to save the captured images:-<br>
<b>line 34: file_name_path = '/Datasets/Train/person/' + str(count) + '.jpg'</b>
4. In line45: Provide the condition for count in order to stop capturing image after a certain count of images.
By default the count will stop at 100 images:-<br>
<b>line 45: if cv2.waitKey(1) == 13 or count == 100:</b><br>
Now run train-test-set-generator.py to generate the dataset of your images

### Collect some Images for testing/validation purposes in 'Datasets/Test' folder

### Train You Model on the generated dataset
1. Open file train-model.py
2. In line 31: Provide the path to your train folder:-<br>
<b>line 31: folders = glob('Datasets/Train/*')</b>
3. Similarly in <b>line 62</b> and <b>line 67</b> Give the path to train and test folder:-<br>
<b>line 62: training_set = train_datagen.flow_from_directory('Datasets/Train',</b><br>
<b>line 62: test_set = train_datagen.flow_from_directory('Datasets/Test',</b>
4. In line 105: Provide the path in which you want to save your model:-<br>
<b>line 105: model.save('Models/facefeatures_new_model.h5')</b><br>
Now the above model will be used in the Face Recognition API to detect your face

### Face Recognition API
1. Open the file face-recog.py
2. In line 15: Provide the path of model generated using previous steps to load the model:-<br>
<b>line 15: model = load_model('Models/facefeatures_new_model.h5')</b>
3. In line 19: Provide the path to HaarCascade Classifier XML File:-<br>
<b>line 19: face_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')</b>
4. In line 59: you have to provide the index of your class inside the train folder to show a certain name on detecting the Face:-<br>
if(pred[0]<b>[0]</b>0.5):  
            name=<b>'Aditya'</b><br>

<b>NOW RUN THE API AND YOU CAMERA WILL POP UP AND IT CAN NOW DETECT YOUR FACE</b>
