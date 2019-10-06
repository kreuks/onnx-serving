import time

import grpc
import numpy as np

import predict_pb2
import onnx_ml_pb2
import prediction_service_pb2_grpc


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

    return request_message


def __parse_response(response) -> dict:
    label = np.frombuffer(response.outputs['label'].raw_data, dtype=np.int64)
    scores = np.frombuffer(response.outputs['probabilities'].raw_data, dtype=np.float32)
    return {'label': label[0], 'score': max(scores)}


def __get_inference(url: str, port_number: str, data: np.array) -> dict:
    with grpc.insecure_channel('{url}:{port_number}'.format(url=url, port_number=port_number)) as channel:
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request_message = __get_request_message(data)
        response = stub.Predict(request_message)
        response = __parse_response(response)
        return response


test_data = np.load('../test_data.npy')
for x in test_data:
    start = time.time()
    print(__get_inference('localhost', '50051', x))
    print('inference time ', (time.time() - start) * 1000)
