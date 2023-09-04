import urllib.request 
from os import path


url = 'https://github.com/microsoft/MS-SNSD/raw/master/clean_test/'
for i in range(200):
    filename = f"clnsp{i}.wav"
    new_url = url + filename
    urllib.request.urlretrieve(new_url, path.join('/home/daria/Pobrane/wav_test', filename))
    