import sys

import onnx
from onnxsim import simplify

if len(sys.argv) == 3:
    onnx_in = sys.argv[1]
    onnx_out = sys.argv[2]

    onnx_model = onnx.load(onnx_in)
    model_simpified, check = simplify(onnx_model)

    if check:
        onnx.save(model_simpified, onnx_out)
        print("The simplified model is at {}".format(onnx_out))
    else:
        print("Simplification failed...")

else:
    print("Usage: arg1->input, arg2->output")
