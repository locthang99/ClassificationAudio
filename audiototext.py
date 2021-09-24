import speech_recognition as sr
r = sr.Recognizer()
file =sr.AudioFile("128.mp3")
with file as src:
    audio = r.record(src)
r.recognize_google(audio)