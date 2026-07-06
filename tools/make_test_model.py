# Genera un model.onnx de prueba (regresion lineal 3 features -> 1 salida) para
# probar la inferencia ONNX REAL del serve Go (build -tags onnx).
# Requiere: pip install scikit-learn skl2onnx onnx
import numpy as np
from sklearn.linear_model import LinearRegression
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3)).astype("float32")
y = (X[:, 0] * 2.0 - X[:, 1] + 0.5 * X[:, 2]).astype("float32")

model = LinearRegression().fit(X, y)
onx = to_onnx(model, initial_types=[("input", FloatTensorType([None, 3]))],
              target_opset={"": 18})
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# Coeficientes reales para verificar la inferencia en Go:
print("model.onnx generado. input='input' [None,3] -> output='variable' [None,1]")
print("coef:", model.coef_, "intercept:", float(model.intercept_))
print("ejemplo: input [1,2,3] ->", float(model.predict([[1.0, 2.0, 3.0]])[0]))
