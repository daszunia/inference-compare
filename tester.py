import requests
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from prometheus_client import start_http_server, Summary

bentoml_text_headers = {
    'accept': 'text/plain',
    'Content-Type': 'text/plain',
}

bentoml_audio_headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'accept': 'text/plain',
}

kseve_text_headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
}

# FRAMEWORKS
CONST_BENTOML = 'bentoml'
CONST_TORCHSERVE = 'torchserve'
CONST_KSERVE = 'kserve'

# MODELS
CONST_VIS = 'vis'
CONST_WAV2VEC = 'wav2vec'
CONST_DIALOGPT = 'dialogpt'

# ADDRESSES
address1 = 'localhost:8004'
address2 = 'localhost:8002'
address3 = 'localhost:8003'
address4 = 'localhost:8080'

# BENTOML PATHS
bentoml_dialogpt = '/dialogpt'
bentoml_vis = '/vis'
bentoml_wav2vec = '/wav2vec'

# TORCHSERVE PATHS
torchserve_dialogpt = '/predictions/dialogpt'
torchserve_vis = '/predictions/vis'
torchserve_wav2vec = '/predictions/wav2vec'

# KSERVE PATHS
kserve_dialogpt = '/v1/models/kserve-custom-dialogpt:predict'
kserve_vis = '/v1/models/kserve-custom-vision:predict'
kserve_wav2vec = '/v1/models/kserve-custom-wav2vec:predict'

# LOCKS
dialogpt_test_data_lock = threading.Lock()
vis_test_data_lock = threading.Lock()
audio_test_data_lock = threading.Lock()
kserve_lock = threading.Lock()
summary_lock = threading.Lock()


class Tester:
    def __init__(self):
        self.requests_summary = Summary('request_latency_miliseconds', 'Time it took to get response from model', ['framework', 'model'])
        self.dialogpt_test_data = self.prepare_dialogpt_test_data()
        self.vis_test_data = self.prepare_vis_test_data()
        self.wav_test_path = './wav_test/clnsp'
        self.image_test_path = '/home/daria/projects/mgr/Im2Text/data/images/'

    def prepare_dialogpt_test_data(self):
        with open("dialogpt-test-data.txt") as f:
            return f.readlines()
    
    def prepare_vis_test_data(self):
        with open("vis-test-data.txt") as f:
            return f.readlines()

    def prepare_address(self, framework, model):
        address = 'http://'
        if framework == CONST_BENTOML and model == CONST_DIALOGPT:
            address += address1
            address += bentoml_dialogpt
        if framework == CONST_BENTOML and model == CONST_VIS:
            address += address2
            address += bentoml_vis
        if framework == CONST_BENTOML and model == CONST_WAV2VEC:
            address += address3
            address += bentoml_wav2vec

        if framework == CONST_TORCHSERVE and model == CONST_DIALOGPT:
            address += address1
            address += torchserve_dialogpt
        if framework == CONST_TORCHSERVE and model == CONST_VIS:
            address += address2
            address += torchserve_vis
        if framework == CONST_TORCHSERVE and model == CONST_WAV2VEC:
            address += address3
            address += torchserve_wav2vec

        if framework == CONST_KSERVE and model == CONST_DIALOGPT:
            address += address4
            address += kserve_dialogpt
        if framework == CONST_KSERVE and model == CONST_VIS:
            address += address4
            address += kserve_vis
        if framework == CONST_KSERVE and model == CONST_WAV2VEC:
            address += address4
            address += kserve_wav2vec
        return address

    def prepare_headers(self, framework, model):
        headers = {}
        if framework == CONST_BENTOML and (model == CONST_DIALOGPT or model == CONST_VIS):
            headers = bentoml_text_headers
        if framework == CONST_BENTOML and model == CONST_WAV2VEC:
            headers = bentoml_audio_headers
        if framework == CONST_KSERVE:
            headers = kseve_text_headers
        return headers

    def prepare_data(self, framework, model):
        data = {}
        idx = random.randint(0, 199)
        if model == CONST_DIALOGPT:
            with dialogpt_test_data_lock:
                data = self.dialogpt_test_data[idx]
        if model == CONST_VIS:
            with vis_test_data_lock:
                data = self.vis_test_data[idx]
                #data = open(self.image_test_path + str(idx) + '.png', 'rb').read()
        if model == CONST_WAV2VEC:
            with audio_test_data_lock:
                data = open(self.wav_test_path + str(idx) + '.wav', 'rb').read()
        if framework == CONST_KSERVE and model == CONST_DIALOGPT:
            with kserve_lock:
                old_data = data
                data = "{\"data\": \"" + old_data.strip() + "\"}"
                print(data)
        return data

    def send_request(self, framework, model):
        address = self.prepare_address(framework, model)
        headers = self.prepare_headers(framework, model)
        data = self.prepare_data(framework, model)

        start = time.perf_counter()
        response = requests.post(address, headers=headers, data=data)
        request_time = time.perf_counter() - start
        print(response.text)
        request_miliseconds = int(request_time * 1000)

        with summary_lock:
            self.requests_summary.labels(framework=framework, model=model).observe(request_miliseconds)
        print(f"Request completed in {request_miliseconds}")


if __name__ == '__main__':
    random.seed(123)
    tester = Tester()
    start_http_server(8000)
    print("starting")
    while True:
        #tester.send_request(CONST_TORCHSERVE, CONST_DIALOGPT)
        tester.send_request(CONST_TORCHSERVE, CONST_VIS)
        tester.send_request(CONST_TORCHSERVE, CONST_WAV2VEC)

