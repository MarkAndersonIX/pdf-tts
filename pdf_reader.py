import os
import platform
import sys
import io
from typing import Optional
import PyPDF2
#import pygame #loops with in memory fp
try:
    import pyttsx3 
except Exception as ex:
    print(f"pyttsx3 was not loaded. {ex}")
from enum import Enum
try:
    from gtts import gTTS
    from pydub import AudioSegment
    from pydub.playback import play
except Exception as ex:
    print(f"gtts libraries were not loaded. {ex}")
    
# pyttsx3 requires apt install espeak on linux
# libstdcxx.so may mismatch.  You can either set LD_PRELOAD to the system library
# export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
# or install lib with conda (probably the better option.)
# conda install -c conda-forge libstdcxx-ng


class Engine(Enum):
    PYTTSX3=1,
    GTTS=2

ENGINE = Engine.GTTS

def getSubfolder(basedir) -> str:
    validEntry = False
    folders = []
    print(f"Please pick a subfolder by number or 'q' to exit, follwed by Enter.\n")
    for root, dirs, files in os.walk(basedir, topdown=False):
        for idx, name in enumerate(dirs):
            folder = os.path.join(root, name)
            print(f"\t{idx}. {folder}")
            folders.append(folder)
    while True:
        response = input("Selection: ")
        if response == 'q':
            exit(0)
        if response.isdigit():
            response = int(response)
            if 0 <= response < len(folders):
                print(f"Folder {folders[response]} selected.")
                return folders[response]
        else:
            print(type(response), response, len(folders))

def getBookPath(basedir) -> str:
    validEntry = False
    paths = []
    print(f"Please pick a book by number or 'q' to exit, follwed by Enter.\n")
    for root, dirs, files in os.walk(basedir, topdown=False):
        for idx, name in enumerate(filter(lambda x: x.endswith(".pdf"), files)):
            path = os.path.join(root, name)
            print(f"\t{idx}. {path}")
            paths.append(path)
    while True:
        response = input("Selection: ")
        if response == 'q':
            exit(0)
        if response.isdigit():
            response = int(response)
            if 0 <= response < len(paths):
                print(f"Book {paths[response]} selected.")
                return paths[response]

#initialize engine and return the resulting object.
def init_engine() -> object:
    if ENGINE == Engine.PYTTSX3:
        # try to initialize pyttsx3
        try:
            play = pyttsx3.init()
            return play  
        except Exception as ex:
            f"Could not load environment. {ex}"
    elif ENGINE == Engine.GTTS:
            return
    else:
        print("unsupported engine")
        exit(0)

def play_data(data: str, engine: Optional[object]):
    if not data:
        return
    if ENGINE == 'PYTTSX3':
        engine.say(data)
        engine.runAndWait()
    else:
        # Generate speech
        tts = gTTS(text=data, lang='en')
        # Save to a file-like object
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio = AudioSegment.from_file(fp, format="mp3")
        play(audio)
        return

system = platform.system()     
if system == 'Linux':
    basedir = "/media/user/Barracuda1TB/Training/Books/"
elif system == 'Windows':
    basedir = 'D:\\Training\\Books\\'
else:
    print(f'Unsupported OS {system}')
engine = init_engine()
subfolder = getSubfolder(basedir)
bookpath = getBookPath(subfolder)
print(f"Loading book {bookpath}...")
pdf_reader = PyPDF2.PdfReader(bookpath)
metadata = pdf_reader.metadata
num_pages = len(pdf_reader.pages)
print(f"metadata: {metadata}, num_pages: {num_pages}")
# file = open('demo.pdf','rb')
print('Playing PDF File..')
for num in range(0,num_pages):
    page = pdf_reader.pages[num]
    data= page.extract_text()
    play_data(data, engine)
