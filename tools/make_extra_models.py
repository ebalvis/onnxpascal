# Genera modelos ONNX de prueba para multi-IO e int64 (con onnx.helper).
#   model_int64.onnx : y = x + 10        (int64, 1 entrada -> 1 salida)
#   model_multi.onnx : out_id = x ; out_dbl = x*2   (float, 1 entrada -> 2 salidas)
# Requiere: pip install onnx numpy
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

# --- int64: y = x + 10 ---
x  = helper.make_tensor_value_info("x", TensorProto.INT64, [None])
y  = helper.make_tensor_value_info("y", TensorProto.INT64, [None])
ten = numpy_helper.from_array(np.array([10], dtype=np.int64), name="ten")
g  = helper.make_graph([helper.make_node("Add", ["x", "ten"], ["y"])],
                       "int64add", [x], [y], [ten])
m  = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
m.ir_version = 9
onnx.save(m, "model_int64.onnx")

# --- multi-salida: out_id = x ; out_dbl = x*2 ---
xf = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])
o1 = helper.make_tensor_value_info("out_id",  TensorProto.FLOAT, [None])
o2 = helper.make_tensor_value_info("out_dbl", TensorProto.FLOAT, [None])
two = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name="two")
g2 = helper.make_graph([helper.make_node("Identity", ["x"], ["out_id"]),
                        helper.make_node("Mul", ["x", "two"], ["out_dbl"])],
                       "multi", [xf], [o1, o2], [two])
m2 = helper.make_model(g2, opset_imports=[helper.make_opsetid("", 18)])
m2.ir_version = 9
onnx.save(m2, "model_multi.onnx")

# --- multi-entrada: y = a + b  (2 entradas float) ---
a  = helper.make_tensor_value_info("a", TensorProto.FLOAT, [None])
b  = helper.make_tensor_value_info("b", TensorProto.FLOAT, [None])
yo = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])
g3 = helper.make_graph([helper.make_node("Add", ["a", "b"], ["y"])],
                       "add2", [a, b], [yo])
m3 = helper.make_model(g3, opset_imports=[helper.make_opsetid("", 18)])
m3.ir_version = 9
onnx.save(m3, "model_add.onnx")

print("model_int64.onnx + model_multi.onnx + model_add.onnx generados")
