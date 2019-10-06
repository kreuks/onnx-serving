import onnxruntime as rt
import numpy as np


test_data = np.load('../test_data.npy')

sess = rt.InferenceSession('../xgboost.onnx')

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
probabilities = sess.get_outputs()[1].name

for x in test_data:
    pred_onx = sess.run([probabilities], {input_name: np.expand_dims(x, axis=0).astype(np.float32)})[0]
    label = np.argmax(pred_onx)
    result = {'label': label, 'probability': max(np.squeeze(pred_onx))}
    print(result)
