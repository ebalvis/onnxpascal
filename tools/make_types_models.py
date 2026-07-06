# Modelos ONNX de prueba para los tipos añadidos en v0.3:
#   model_f64.onnx : y = x + 1.0   (DOUBLE / float64)
#   model_i32.onnx : y = x + 1     (INT32)
#   model_u8.onnx  : y = x         (UINT8, Identity)
# Requiere: pip install onnx numpy
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

# --- float64: y = x + 1.0 ---
x  = helper.make_tensor_value_info("x", TensorProto.DOUBLE, [None])
y  = helper.make_tensor_value_info("y", TensorProto.DOUBLE, [None])
one = numpy_helper.from_array(np.array([1.0], dtype=np.float64), name="one")
g  = helper.make_graph([helper.make_node("Add", ["x", "one"], ["y"])],
                       "f64add", [x], [y], [one])
m  = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]); m.ir_version = 9
onnx.save(m, "model_f64.onnx")

# --- int32: y = x + 1 ---
xi = helper.make_tensor_value_info("x", TensorProto.INT32, [None])
yi = helper.make_tensor_value_info("y", TensorProto.INT32, [None])
oni = numpy_helper.from_array(np.array([1], dtype=np.int32), name="one_i")
g2 = helper.make_graph([helper.make_node("Add", ["x", "one_i"], ["y"])],
                       "i32add", [xi], [yi], [oni])
m2 = helper.make_model(g2, opset_imports=[helper.make_opsetid("", 18)]); m2.ir_version = 9
onnx.save(m2, "model_i32.onnx")

# --- uint8: y = x (Identity) ---
xu = helper.make_tensor_value_info("x", TensorProto.UINT8, [None])
yu = helper.make_tensor_value_info("y", TensorProto.UINT8, [None])
g3 = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])],
                       "u8id", [xu], [yu])
m3 = helper.make_model(g3, opset_imports=[helper.make_opsetid("", 18)]); m3.ir_version = 9
onnx.save(m3, "model_u8.onnx")

print("model_f64.onnx + model_i32.onnx + model_u8.onnx generados")
