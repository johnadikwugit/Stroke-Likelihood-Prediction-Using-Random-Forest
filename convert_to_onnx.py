import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

with open("trained_model.sav", "rb") as f:
    model = pickle.load(f)

n_features = 10
onx = convert_sklearn(model, initial_types=[("input", FloatTensorType([None, n_features]))])

with open("stroke_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

print("stroke_model.onnx created successfully.")
