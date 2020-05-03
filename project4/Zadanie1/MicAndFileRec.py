import speech_recognition as sr
import sys

r = sr.Recognizer()
mic = sr.Microphone(device_index=0)
with mic as source:
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)
v = r.recognize_google(audio)
print(v)

if (len(sys.argv)>1):
    hello = sr.AudioFile(str(sys.argv[1]))
    with hello as source1:
        audio1 = r.record(source1)
    u = r.recognize_google(audio1)
    print(u)
