import time
import requests

import numpy as np
from google.protobuf.json_format import MessageToJson

import predict_pb2
import onnx_ml_pb2


def __get_request_message(data: np.array) -> predict_pb2.PredictRequest:
    input_np_array = np.array(data, dtype=np.float32)
    input_np_array = np.expand_dims(input_np_array, axis=0)

    input_tensor = onnx_ml_pb2.TensorProto()
    input_tensor.dims.extend(input_np_array.shape)
    input_tensor.data_type = 1  # float
    input_tensor.raw_data = input_np_array.tobytes()

    request_message = predict_pb2.PredictRequest()
    request_message.inputs['float_input'].data_type = input_tensor.data_type
    request_message.inputs['float_input'].dims.extend(input_np_array.shape)
    request_message.inputs['float_input'].raw_data = input_tensor.raw_data
    request_message = MessageToJson(request_message)

    return request_message


def __get_request_header() -> dict:
    return {
        'Content-Type': 'application/json',
        'Accept': 'application/x-protobuf'
    }


def __parse_response(response) -> dict:
    label = np.frombuffer(response.outputs['label'].raw_data, dtype=np.int64)
    scores = np.frombuffer(response.outputs['probabilities'].raw_data, dtype=np.float32)
    return {'label': label[0], 'score': max(scores)}


def __get_inference(url: str, port_number: str, data: np.array) -> dict:
    inference_url = '{url}:{port_number}/v1/models/mymodel/versions/1:predict'.format(url=url, port_number=port_number)
    header = __get_request_header()
    data = __get_request_message(data)
    response = requests.post(inference_url, headers=header, data=data)

    response_message = predict_pb2.PredictResponse()
    response_message.ParseFromString(response.content)

    return __parse_response(response_message)


test_data = np.load('../test_data.npy')
for x in test_data:
    start = time.time()
    print(__get_inference('http://localhost', '9001', x))
    print('inference time ', (time.time() - start) * 1000)
