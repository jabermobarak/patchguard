"""Export the learned scoring head, with numerical parity checked separately."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .features import GRID
from .model import Detector


def export_onnx(model: Detector, path: str | Path) -> None:
    f = model.mean.shape[-1]
    constants = [
        numpy_helper.from_array(model.mean, "mean"),
        numpy_helper.from_array(model.precision, "precision"),
        numpy_helper.from_array(np.array([-2], np.int64), "insert_axis"),
        numpy_helper.from_array(np.array([-1], np.int64), "sum_axis"),
        numpy_helper.from_array(np.array(0, np.float32), "zero"),
    ]
    nodes = [
        helper.make_node("Sub", ["features", "mean"], ["d"]),
        helper.make_node("Unsqueeze", ["d", "insert_axis"], ["row"]),
        helper.make_node("MatMul", ["row", "precision"], ["weighted_row"]),
        helper.make_node("Squeeze", ["weighted_row", "insert_axis"], ["weighted"]),
        helper.make_node("Mul", ["d", "weighted"], ["products"]),
        helper.make_node("ReduceSum", ["products", "sum_axis"], ["squared"], keepdims=0),
        helper.make_node("Max", ["squared", "zero"], ["positive"]),
        helper.make_node("Sqrt", ["positive"], ["patch_scores"]),
    ]
    graph = helper.make_graph(
        nodes,
        "patchguard-spatial-gaussian",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", GRID, GRID, f])],
        [helper.make_tensor_value_info("patch_scores", TensorProto.FLOAT, ["batch", GRID, GRID])],
        constants,
    )
    result = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    result.ir_version = 9
    onnx.checker.check_model(result)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(result, str(path))
