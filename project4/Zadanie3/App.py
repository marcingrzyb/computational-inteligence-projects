from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import logging
logging.getLogger('tensorflow').disabled = True

from keras.models import load_model

import numpy as np
import sys
import time
import os

import sounddevice as sd
import librosa
from PIL import Image
from gtts import gTTS
from playsound import playsound

filePath = sys.argv[1]
model = load_model('best_model.hdf5')#model with 80% acc, trained 6 epochs. Created using notebook Speech Recognition.ipynb
classes = ['left', 'no', 'right', 'stop', 'yes']
# text to speech prep
language = 'en'
counter = 0

samplerate = 16000
duration = 1  # seconds

direction_map = {"left": 90, "right": -90}


def createMp3(mytext):
    global counter
    counter += 1
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("./records/" + str(counter) + ".mp3")


def createMP3Responses():
    createMp3("Hello, would you like to rotate image?")
    createMp3("Do you want to rotate right,left or to stop the program?")
    createMp3("Do you want to save changes?")
    createMp3("Saved to file edited.jpg, goodbye")
    createMp3("Goodbye")
    createMp3("Unknown command, please repeat")


def prepareEnv():
    if not os.path.isdir('./records'):
        os.mkdir('./records')
    createMP3Responses()


def predict(audio):
    prob = model.predict(audio.reshape(-1, 8000, 1))
    index = np.argmax(prob[0])
    return classes[index]


def getCommand(text):
    playsound("./records/" + str(text) + ".mp3")
    for i in reversed(range(0, 3)):
        print(i + 1)
        time.sleep(1)

    print("recording started")
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
                    channels=1, blocking=True)
    print("recording end")
    sd.wait()

    samples = librosa.resample(mydata.T, samplerate, 8000)
    print(predict(samples))
    return predict(samples)


def imageSave(image):
    cmd = getCommand(3)
    if (cmd == "yes"):
        image.save("./edited.jpg")
        playsound("./records/4.mp3")
    elif (cmd == "no"):
        playsound("./records/5.mp3")
    else:
        imageSave(image)


def rotate(image):
    cmd = getCommand(2)
    if (cmd in direction_map.keys()):
        image = image.rotate(direction_map[cmd])
        image.show()
        time.sleep(3)
        imageSave(image)
    elif (cmd == "stop"):
        playsound("./records/5.mp3")
    else:
        playsound("./records/6.mp3")
        rotate(image)


def application(image):
    command = getCommand(1)
    if (command == "no"):
        image.show()
    elif (command == "yes"):
        imageCp = image
        rotate(imageCp)
    else:
        playsound("./records/6.mp3")
        application(image)


prepareEnv()
image = Image.open(filePath)
application(image)
