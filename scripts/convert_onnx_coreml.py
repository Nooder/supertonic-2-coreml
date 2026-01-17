#!/usr/bin/env python3
import argparse
import math
import os
import sys
from typing import Dict, List

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


class NameGen:
    def __init__(self, model: onnx.ModelProto):
        used = set()
        for init in model.graph.initializer:
            used.add(init.name)
        for node in model.graph.node:
            if node.name:
                used.add(node.name)
            used.update(node.input)
            used.update(node.output)
        for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
            used.add(vi.name)
        self.used = used
        self.counter = 0

    def new(self, base: str) -> str:
        name = base
        while name in self.used:
            self.counter += 1
            name = f"{base}_{self.counter}"
        self.used.add(name)
        return name


def add_const(model: onnx.ModelProto, name_gen: NameGen, base: str, value, dtype=np.float32) -> str:
    name = name_gen.new(base)
    arr = np.array(value, dtype=dtype)
    tensor = numpy_helper.from_array(arr, name=name)
    model.graph.initializer.append(tensor)
    return name


def get_attr(node: onnx.NodeProto, name: str, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    return default


def find_const_value(model: onnx.ModelProto, name: str):
    if not name:
        return None
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] == name:
            for attr in node.attribute:
                if attr.name == "value":
                    return numpy_helper.to_array(attr.t)
    return None


def rewrite_layernorm(model: onnx.ModelProto, name_gen: NameGen) -> int:
    new_nodes: List[onnx.NodeProto] = []
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "LayerNormalization":
            new_nodes.append(node)
            continue
        replaced += 1
        prefix = node.name or node.output[0]
        x = node.input[0]
        scale = node.input[1]
        bias = node.input[2]
        axis = int(get_attr(node, "axis", -1))
        epsilon = float(get_attr(node, "epsilon", 1e-5))

        eps_name = add_const(model, name_gen, f"{prefix}_epsilon", epsilon, np.float32)

        mean = name_gen.new(f"{prefix}_mean")
        centered = name_gen.new(f"{prefix}_centered")
        sq = name_gen.new(f"{prefix}_sq")
        var = name_gen.new(f"{prefix}_var")
        var_eps = name_gen.new(f"{prefix}_var_eps")
        std = name_gen.new(f"{prefix}_std")
        norm = name_gen.new(f"{prefix}_norm")
        scaled = name_gen.new(f"{prefix}_scaled")

        new_nodes.extend(
            [
                helper.make_node("ReduceMean", [x], [mean], axes=[axis], keepdims=1, name=name_gen.new(f"{prefix}_rm")),
                helper.make_node("Sub", [x, mean], [centered], name=name_gen.new(f"{prefix}_sub")),
                helper.make_node("Mul", [centered, centered], [sq], name=name_gen.new(f"{prefix}_sqnode")),
                helper.make_node("ReduceMean", [sq], [var], axes=[axis], keepdims=1, name=name_gen.new(f"{prefix}_rm2")),
                helper.make_node("Add", [var, eps_name], [var_eps], name=name_gen.new(f"{prefix}_vareps")),
                helper.make_node("Sqrt", [var_eps], [std], name=name_gen.new(f"{prefix}_sqrt")),
                helper.make_node("Div", [centered, std], [norm], name=name_gen.new(f"{prefix}_div")),
                helper.make_node("Mul", [norm, scale], [scaled], name=name_gen.new(f"{prefix}_mul")),
                helper.make_node("Add", [scaled, bias], [node.output[0]], name=name_gen.new(f"{prefix}_out")),
            ]
        )
    if replaced:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return replaced


def rewrite_erf(model: onnx.ModelProto, name_gen: NameGen) -> int:
    new_nodes: List[onnx.NodeProto] = []
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "Erf":
            new_nodes.append(node)
            continue
        replaced += 1
        prefix = node.name or node.output[0]
        x = node.input[0]
        y = node.output[0]

        # Abramowitz and Stegun approximation
        p = add_const(model, name_gen, f"{prefix}_p", 0.3275911, np.float32)
        a1 = add_const(model, name_gen, f"{prefix}_a1", 0.254829592, np.float32)
        a2 = add_const(model, name_gen, f"{prefix}_a2", -0.284496736, np.float32)
        a3 = add_const(model, name_gen, f"{prefix}_a3", 1.421413741, np.float32)
        a4 = add_const(model, name_gen, f"{prefix}_a4", -1.453152027, np.float32)
        a5 = add_const(model, name_gen, f"{prefix}_a5", 1.061405429, np.float32)
        one = add_const(model, name_gen, f"{prefix}_one", 1.0, np.float32)

        sign = name_gen.new(f"{prefix}_sign")
        absx = name_gen.new(f"{prefix}_abs")
        px = name_gen.new(f"{prefix}_px")
        denom = name_gen.new(f"{prefix}_denom")
        t = name_gen.new(f"{prefix}_t")
        sq = name_gen.new(f"{prefix}_sq")
        neg_sq = name_gen.new(f"{prefix}_neg_sq")
        exp = name_gen.new(f"{prefix}_exp")

        poly1 = name_gen.new(f"{prefix}_poly1")
        poly2 = name_gen.new(f"{prefix}_poly2")
        poly3 = name_gen.new(f"{prefix}_poly3")
        poly4 = name_gen.new(f"{prefix}_poly4")
        poly5 = name_gen.new(f"{prefix}_poly5")
        poly6 = name_gen.new(f"{prefix}_poly6")
        poly7 = name_gen.new(f"{prefix}_poly7")
        poly8 = name_gen.new(f"{prefix}_poly8")
        poly9 = name_gen.new(f"{prefix}_poly9")

        mul_poly = name_gen.new(f"{prefix}_mul_poly")
        y_pos = name_gen.new(f"{prefix}_y_pos")

        new_nodes.extend(
            [
                helper.make_node("Sign", [x], [sign], name=name_gen.new(f"{prefix}_signnode")),
                helper.make_node("Abs", [x], [absx], name=name_gen.new(f"{prefix}_absnode")),
                helper.make_node("Mul", [p, absx], [px], name=name_gen.new(f"{prefix}_pxnode")),
                helper.make_node("Add", [one, px], [denom], name=name_gen.new(f"{prefix}_denomnode")),
                helper.make_node("Div", [one, denom], [t], name=name_gen.new(f"{prefix}_tnode")),
                helper.make_node("Mul", [absx, absx], [sq], name=name_gen.new(f"{prefix}_sqnode")),
                helper.make_node("Neg", [sq], [neg_sq], name=name_gen.new(f"{prefix}_negsqnode")),
                helper.make_node("Exp", [neg_sq], [exp], name=name_gen.new(f"{prefix}_expnode")),
                helper.make_node("Mul", [a5, t], [poly1], name=name_gen.new(f"{prefix}_p1")),
                helper.make_node("Add", [poly1, a4], [poly2], name=name_gen.new(f"{prefix}_p2")),
                helper.make_node("Mul", [poly2, t], [poly3], name=name_gen.new(f"{prefix}_p3")),
                helper.make_node("Add", [poly3, a3], [poly4], name=name_gen.new(f"{prefix}_p4")),
                helper.make_node("Mul", [poly4, t], [poly5], name=name_gen.new(f"{prefix}_p5")),
                helper.make_node("Add", [poly5, a2], [poly6], name=name_gen.new(f"{prefix}_p6")),
                helper.make_node("Mul", [poly6, t], [poly7], name=name_gen.new(f"{prefix}_p7")),
                helper.make_node("Add", [poly7, a1], [poly8], name=name_gen.new(f"{prefix}_p8")),
                helper.make_node("Mul", [poly8, t], [poly9], name=name_gen.new(f"{prefix}_p9")),
                helper.make_node("Mul", [poly9, exp], [mul_poly], name=name_gen.new(f"{prefix}_mulpoly")),
                helper.make_node("Sub", [one, mul_poly], [y_pos], name=name_gen.new(f"{prefix}_ypos")),
                helper.make_node("Mul", [sign, y_pos], [y], name=name_gen.new(f"{prefix}_out")),
            ]
        )
    if replaced:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return replaced


def rewrite_sin(model: onnx.ModelProto, name_gen: NameGen) -> int:
    new_nodes: List[onnx.NodeProto] = []
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "Sin":
            new_nodes.append(node)
            continue
        replaced += 1
        prefix = node.name or node.output[0]
        x = node.input[0]
        y = node.output[0]

        half_pi = add_const(model, name_gen, f"{prefix}_half_pi", float(math.pi / 2.0), np.float32)
        shifted = name_gen.new(f"{prefix}_shifted")

        new_nodes.extend(
            [
                helper.make_node("Sub", [x, half_pi], [shifted], name=name_gen.new(f"{prefix}_sub")),
                helper.make_node("Cos", [shifted], [y], name=name_gen.new(f"{prefix}_cos")),
            ]
        )
    if replaced:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return replaced


def rewrite_prelu(model: onnx.ModelProto, name_gen: NameGen) -> int:
    new_nodes: List[onnx.NodeProto] = []
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "PRelu":
            new_nodes.append(node)
            continue
        alpha_val = None
        if len(node.input) >= 2:
            alpha_val = find_const_value(model, node.input[1])
        if alpha_val is None:
            new_nodes.append(node)
            continue
        alpha_scalar = float(np.array(alpha_val).flatten()[0])
        prefix = node.name or node.output[0]
        x = node.input[0]
        y = node.output[0]

        alpha_const = add_const(model, name_gen, f"{prefix}_alpha", alpha_scalar, np.float32)
        relu_out = name_gen.new(f"{prefix}_relu")
        neg_out = name_gen.new(f"{prefix}_neg")
        scaled_out = name_gen.new(f"{prefix}_scaled")

        new_nodes.extend(
            [
                helper.make_node("Relu", [x], [relu_out], name=name_gen.new(f"{prefix}_relu")),
                helper.make_node("Sub", [x, relu_out], [neg_out], name=name_gen.new(f"{prefix}_sub")),
                helper.make_node("Mul", [neg_out, alpha_const], [scaled_out], name=name_gen.new(f"{prefix}_mul")),
                helper.make_node("Add", [relu_out, scaled_out], [y], name=name_gen.new(f"{prefix}_add")),
            ]
        )
        replaced += 1
    if replaced:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return replaced


def externalize_text_embedding(model: onnx.ModelProto, max_text_len: int, input_name: str = "text_embed") -> bool:
    gather_node = None
    for node in model.graph.node:
        if node.op_type == "Gather" and "text_ids" in node.input:
            gather_node = node
            break
    if gather_node is None:
        return False

    gather_out = gather_node.output[0]
    transpose_node = None
    for node in model.graph.node:
        if node.op_type == "Transpose" and gather_out in node.input:
            transpose_node = node
            break
    if transpose_node is None:
        return False

    weight_name = gather_node.input[0]
    weight_init = None
    for init in model.graph.initializer:
        if init.name == weight_name:
            weight_init = init
            break
    if weight_init is None or len(weight_init.dims) < 2:
        return False

    embed_dim = int(weight_init.dims[1])
    input_shape = [1, max_text_len, embed_dim]

    perm = None
    for attr in transpose_node.attribute:
        if attr.name == "perm":
            perm = list(attr.ints)
            break
    if perm is None:
        perm = list(reversed(range(len(input_shape))))
    output_shape = [input_shape[i] for i in perm]

    # Replace transpose output usage with new input name
    old_out = transpose_node.output[0]
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == old_out:
                node.input[i] = input_name

    # Remove gather/transpose nodes
    new_nodes = [n for n in model.graph.node if n not in (gather_node, transpose_node)]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    # Remove text_ids input
    new_inputs = [i for i in model.graph.input if i.name != "text_ids"]
    del model.graph.input[:]
    model.graph.input.extend(new_inputs)

    # Add new embedding input
    new_input = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, output_shape)
    model.graph.input.append(new_input)

    # Drop embedding weights initializer
    new_inits = [init for init in model.graph.initializer if init.name != weight_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)
    return True


def rewrite_reshape_dynamic(model: onnx.ModelProto, shapes: Dict[str, List[int]]) -> int:
    replaced = 0
    init_map = {init.name: init for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        if len(node.input) < 2:
            continue
        shape_name = node.input[1]
        if shape_name not in init_map:
            continue
        input_shape = shapes.get(node.input[0])
        if not input_shape:
            continue
        init = init_map[shape_name]
        shape_arr = numpy_helper.to_array(init).astype(np.int64).flatten()
        if (shape_arr == -1).sum() != 1:
            continue
        total = int(np.prod(input_shape))
        known = int(np.prod([int(d) for d in shape_arr.tolist() if d != -1]))
        if known <= 0 or total % known != 0:
            continue
        missing = int(total // known)
        new_shape = [missing if d == -1 else int(d) for d in shape_arr.tolist()]
        new_name = f"{shape_name}_resolved_{replaced}"
        new_init = numpy_helper.from_array(np.array(new_shape, dtype=np.int64), name=new_name)
        model.graph.initializer.append(new_init)
        node.input[1] = new_name
        replaced += 1
    return replaced


def rewrite_pad(model: onnx.ModelProto) -> int:
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "Pad":
            continue
        pads_val = None
        value_val = None
        if len(node.input) >= 2:
            pads_val = find_const_value(model, node.input[1])
        if len(node.input) >= 3:
            value_val = find_const_value(model, node.input[2])
        if pads_val is None:
            continue
        pads_list = [int(x) for x in pads_val.flatten().tolist()]

        attrs = [a for a in node.attribute if a.name not in ("pads", "value")]
        attrs.append(helper.make_attribute("pads", pads_list))
        if value_val is not None:
            attrs.append(helper.make_attribute("value", float(np.array(value_val).flatten()[0])))
        del node.attribute[:]
        node.attribute.extend(attrs)
        node.input[:] = [node.input[0]]
        replaced += 1
    return replaced


def rewrite_axes_input_to_attr(model: onnx.ModelProto, op_type: str) -> int:
    replaced = 0
    for node in model.graph.node:
        if node.op_type != op_type:
            continue
        if len(node.input) < 2:
            continue
        axes_val = find_const_value(model, node.input[1])
        if axes_val is None:
            continue
        axes_list = [int(x) for x in np.array(axes_val).flatten().tolist()]
        attrs = [a for a in node.attribute if a.name != "axes"]
        attrs.append(helper.make_attribute("axes", axes_list))
        del node.attribute[:]
        node.attribute.extend(attrs)
        node.input[:] = [node.input[0]]
        replaced += 1
    return replaced


def rewrite_split(model: onnx.ModelProto) -> int:
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "Split":
            continue
        if len(node.input) < 2:
            continue
        split_val = find_const_value(model, node.input[1])
        if split_val is None:
            continue
        split_list = [int(x) for x in np.array(split_val).flatten().tolist()]
        attrs = [a for a in node.attribute if a.name != "split"]
        attrs.append(helper.make_attribute("split", split_list))
        del node.attribute[:]
        node.attribute.extend(attrs)
        node.input[:] = [node.input[0]]
        replaced += 1
    return replaced


def rewrite_clip_inputs(model: onnx.ModelProto) -> int:
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "Clip":
            continue
        while node.input and node.input[-1] == "":
            node.input.pop()
            replaced += 1
    return replaced


def rewrite_clip_int_to_float(model: onnx.ModelProto, name_gen: NameGen) -> int:
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        return 0

    types = get_onnx_tensor_types(inferred)
    new_nodes: List[onnx.NodeProto] = []
    replaced = 0

    for node in model.graph.node:
        if node.op_type != "Clip":
            new_nodes.append(node)
            continue
        x_name = node.input[0] if node.input else None
        x_type = types.get(x_name)
        if x_type not in (TensorProto.INT32, TensorProto.INT64):
            new_nodes.append(node)
            continue

        cast_in_name = name_gen.new(f"{node.name or 'Clip'}_cast_in")
        cast_in_out = name_gen.new(f"{node.name or 'Clip'}_x_float")
        clip_out_tmp = name_gen.new(f"{node.name or node.output[0]}_float")
        cast_out_name = name_gen.new(f"{node.name or 'Clip'}_cast_out")

        new_nodes.append(
            helper.make_node(
                "Cast",
                [x_name],
                [cast_in_out],
                name=cast_in_name,
                to=TensorProto.FLOAT,
            )
        )

        clip_node = helper.make_node(
            "Clip",
            [cast_in_out] + list(node.input[1:]),
            [clip_out_tmp],
            name=node.name,
        )
        clip_node.attribute.extend(node.attribute)
        new_nodes.append(clip_node)

        new_nodes.append(
            helper.make_node(
                "Cast",
                [clip_out_tmp],
                [node.output[0]],
                name=cast_out_name,
                to=x_type,
            )
        )

        replaced += 1

    if replaced:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return replaced


def set_opset(model: onnx.ModelProto, target: int) -> None:
    del model.opset_import[:]
    model.opset_import.append(helper.make_opsetid("", target))


def clear_value_info(model: onnx.ModelProto) -> None:
    del model.graph.value_info[:]


def infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        return onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        print(f"- shape inference failed: {exc}")
        return model


def simplify_model(model: onnx.ModelProto, input_shapes: Dict[str, List[int]]) -> onnx.ModelProto:
    try:
        from onnxsim import simplify
    except Exception as exc:
        print(f"onnxsim not available: {exc}")
        return model
    model_s, check = simplify(model, input_shapes=input_shapes)
    if not check:
        print("Warning: onnxsim reported the simplified model may be invalid")
    return model_s


def get_supported_ops() -> set:
    try:
        import onnx_coreml  # type: ignore
        import inspect
        import os as _os
        import re as _re
        base_dir = _os.path.dirname(onnx_coreml.__file__)
        pattern = _re.compile(r"\s*\"([A-Za-z0-9_]+)\"\s*:\s*_convert_")
        ops = set()
        for fname in ("_operators.py", "_operators_nd.py"):
            path = _os.path.join(base_dir, fname)
            if not _os.path.exists(path):
                continue
            text = open(path, "r", encoding="utf-8").read()
            ops.update(pattern.findall(text))
        return ops
    except Exception:
        return set()


def ensure_nnssa_stub() -> None:
    try:
        import coremltools.converters.nnssa.coreml.graph_pass.mlmodel_passes  # type: ignore
        return
    except Exception:
        import types
        import sys as _sys

        def _mkpkg(name: str) -> types.ModuleType:
            mod = types.ModuleType(name)
            mod.__path__ = []
            return mod

        _sys.modules.setdefault("coremltools.converters.nnssa", _mkpkg("coremltools.converters.nnssa"))
        _sys.modules.setdefault("coremltools.converters.nnssa.coreml", _mkpkg("coremltools.converters.nnssa.coreml"))
        _sys.modules.setdefault(
            "coremltools.converters.nnssa.coreml.graph_pass",
            _mkpkg("coremltools.converters.nnssa.coreml.graph_pass"),
        )
        passes = types.ModuleType("coremltools.converters.nnssa.coreml.graph_pass.mlmodel_passes")
        passes.remove_disconnected_layers = lambda spec: spec
        passes.transform_conv_crop = lambda spec: spec
        _sys.modules["coremltools.converters.nnssa.coreml.graph_pass.mlmodel_passes"] = passes


def patch_coreml_expand_dims() -> None:
    try:
        from coremltools.models.neural_network import builder as nn_builder

        if getattr(nn_builder.NeuralNetworkBuilder.add_expand_dims, "_patched", False):
            return

        def _patched_add_expand_dims(self, name, input_name, output_name, axes):
            spec_layer = self._add_generic_layer(name, [input_name], [output_name])
            spec_layer_params = spec_layer.expandDims
            spec_layer_params.axes.extend(axes)
            rank = self._get_rank(input_name)
            if rank is None:
                rank = 0
            self.rank_dict[output_name] = rank + len(axes)
            return spec_layer

        _patched_add_expand_dims._patched = True  # type: ignore[attr-defined]
        nn_builder.NeuralNetworkBuilder.add_expand_dims = _patched_add_expand_dims  # type: ignore[assignment]
    except Exception:
        return


def patch_onnx_coreml_dynamic_inputs() -> None:
    try:
        import onnx_coreml._operators_nd as ops  # type: ignore
        import numpy as _np

        if getattr(ops, "_patched_dynamic_inputs", False):
            return

        def _axes_from_input(node):
            if len(node.inputs) > 1 and node.inputs[1] in node.input_tensors:
                axes_val = node.input_tensors[node.inputs[1]]
                if isinstance(axes_val, _np.ndarray):
                    return [int(x) for x in axes_val.flatten().tolist()]
                if isinstance(axes_val, list):
                    return [int(x) for x in axes_val]
            return None

        def _convert_unsqueeze(builder, node, graph, err):
            axes = node.attrs.get("axes")
            if axes is None:
                axes = _axes_from_input(node)
            builder.add_expand_dims(
                name=node.name,
                input_name=node.inputs[0],
                output_name=node.outputs[0],
                axes=axes,
            )

        def _convert_squeeze(builder, node, graph, err):
            axes = node.attrs.get("axes")
            if axes is None:
                axes = _axes_from_input(node)
            builder.add_squeeze(
                name=node.name,
                input_name=node.inputs[0],
                output_name=node.outputs[0],
                axes=axes,
            )

        def _convert_pad(builder, node, graph, err):
            mode = node.attrs.get("mode", "constant")
            try:
                mode = mode.decode()
            except (UnicodeDecodeError, AttributeError):
                pass

            pads = node.attrs.get("pads", [])
            if (pads is None or len(pads) == 0) and len(node.inputs) > 1 and node.inputs[1] in node.input_tensors:
                pads_val = node.input_tensors[node.inputs[1]]
                if isinstance(pads_val, _np.ndarray):
                    pads = [int(x) for x in pads_val.flatten().tolist()]
                else:
                    pads = [int(x) for x in pads_val]

            value = node.attrs.get("value", 0.0)
            if len(node.inputs) > 2 and node.inputs[2] in node.input_tensors:
                val = node.input_tensors[node.inputs[2]]
                if isinstance(val, _np.ndarray):
                    value = float(val.flatten()[0])
                else:
                    value = float(val)

            if not pads:
                err.unsupported_op_configuration(builder, node, graph, "Pads not found")

            rank = len(pads) // 2
            begin = pads[:rank]
            end = pads[rank:]

            if mode in ("edge", "reflect"):
                if rank in (3, 4) and all(x == 0 for x in begin[:-2]) and all(x == 0 for x in end[:-2]):
                    top = begin[-2] if rank >= 2 else 0
                    bottom = end[-2] if rank >= 2 else 0
                    left = begin[-1] if rank >= 1 else 0
                    right = end[-1] if rank >= 1 else 0
                    pad_type = "replication" if mode == "edge" else "reflection"
                    builder.add_padding(
                        name=node.name,
                        input_name=node.inputs[0],
                        output_name=node.outputs[0],
                        padding_type=pad_type,
                        top=top,
                        bottom=bottom,
                        left=left,
                        right=right,
                        value=value,
                    )
                    return
                err.unsupported_op_configuration(builder, node, graph, f"Pad mode {mode} unsupported for rank {rank}")

            if mode != "constant":
                err.unsupported_op_configuration(builder, node, graph, "Only constant/edge/reflect Pad supported")

            builder.add_constant_pad(
                name=node.name,
                input_names=[node.inputs[0]],
                output_name=node.outputs[0],
                value=value,
                pad_to_given_output_size_mode=False,
                pad_amounts=pads,
            )

        ops._convert_unsqueeze = _convert_unsqueeze
        ops._convert_squeeze = _convert_squeeze
        ops._convert_pad = _convert_pad
        if hasattr(ops, "_ONNX_NODE_REGISTRY_ND"):
            ops._ONNX_NODE_REGISTRY_ND["Unsqueeze"] = _convert_unsqueeze
            ops._ONNX_NODE_REGISTRY_ND["Squeeze"] = _convert_squeeze
            ops._ONNX_NODE_REGISTRY_ND["Pad"] = _convert_pad
        ops._patched_dynamic_inputs = True
    except Exception:
        return


def list_unsupported_ops(model: onnx.ModelProto, supported: set) -> List[str]:
    ops = sorted({n.op_type for n in model.graph.node})
    if not supported:
        return []
    return sorted([o for o in ops if o not in supported])


def get_onnx_tensor_shapes(model: onnx.ModelProto) -> Dict[str, List[int]]:
    shapes: Dict[str, List[int]] = {}

    def add_value_info(vi: onnx.ValueInfoProto) -> None:
        if not vi.type.HasField("tensor_type"):
            return
        t = vi.type.tensor_type
        if not t.HasField("shape"):
            return
        dims: List[int] = []
        for dim in t.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            else:
                return
        shapes[vi.name] = dims

    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        add_value_info(vi)
    return shapes


def get_onnx_tensor_types(model: onnx.ModelProto) -> Dict[str, int]:
    types: Dict[str, int] = {}

    def add_value_info(vi: onnx.ValueInfoProto) -> None:
        if not vi.type.HasField("tensor_type"):
            return
        t = vi.type.tensor_type
        if t.elem_type:
            types[vi.name] = t.elem_type

    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        add_value_info(vi)
    for init in model.graph.initializer:
        if init.data_type:
            types[init.name] = init.data_type
    return types


def patch_prelu_input_tensors(mlmodel_path: str, onnx_shapes: Dict[str, List[int]]) -> int:
    try:
        from coremltools.models.utils import load_spec, save_spec  # type: ignore
    except Exception as exc:
        print(f"- skip PReLU inputTensor patch: {exc}")
        return 0

    spec = load_spec(mlmodel_path)
    if not spec.HasField("neuralNetwork"):
        return 0

    patched = 0
    for layer in spec.neuralNetwork.layers:
        if layer.WhichOneof("layer") != "activation":
            continue
        if layer.activation.WhichOneof("NonlinearityType") != "PReLU":
            continue
        if not layer.input:
            continue
        shape = onnx_shapes.get(layer.input[0])
        if not shape:
            continue
        # CoreML activation PReLU expects rank >= 3. Map ONNX shapes (N, C, L/...) to CoreML (C, H, W).
        coreml_shape = list(shape)
        if coreml_shape and coreml_shape[0] == 1:
            coreml_shape = coreml_shape[1:]
        if len(coreml_shape) == 1:
            coreml_shape = [coreml_shape[0], 1, 1]
        elif len(coreml_shape) == 2:
            coreml_shape = [coreml_shape[0], 1, coreml_shape[1]]
        if len(coreml_shape) < 3:
            continue
        del layer.inputTensor[:]
        tensor = layer.inputTensor.add()
        tensor.rank = len(coreml_shape)
        tensor.dimValue.extend(coreml_shape)
        patched += 1

    if patched:
        save_spec(spec, mlmodel_path)
        print(f"- patched PReLU inputTensor: {patched}")
    return patched


def load_cfgs(cfg_dir: str) -> dict:
    cfg_path = os.path.join(cfg_dir, "tts.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return __import__("json").load(f)


def build_input_shapes(cfgs: dict, max_text_len: int, max_seconds: float) -> Dict[str, Dict[str, List[int]]]:
    sample_rate = cfgs["ae"]["sample_rate"]
    base_chunk_size = cfgs["ae"]["base_chunk_size"]
    chunk_compress = cfgs["ttl"]["chunk_compress_factor"]
    latent_dim = cfgs["ttl"]["latent_dim"]

    latent_channels = latent_dim * chunk_compress
    latent_len = int(math.ceil((max_seconds * sample_rate) / (base_chunk_size * chunk_compress)))

    shapes = {
        "duration_predictor.onnx": {
            "text_ids": [1, max_text_len],
            "style_dp": [1, 8, 16],
            "text_mask": [1, 1, max_text_len],
        },
        "text_encoder.onnx": {
            "text_ids": [1, max_text_len],
            "style_ttl": [1, 50, 256],
            "text_mask": [1, 1, max_text_len],
        },
        "vector_estimator.onnx": {
            "noisy_latent": [1, latent_channels, latent_len],
            "text_emb": [1, 256, max_text_len],
            "style_ttl": [1, 50, 256],
            "latent_mask": [1, 1, latent_len],
            "text_mask": [1, 1, max_text_len],
            "current_step": [1],
            "total_step": [1],
        },
        "vocoder.onnx": {
            "latent": [1, latent_channels, latent_len],
        },
    }
    return shapes


def convert_one(path: str, out_dir: str, input_shapes: Dict[str, List[int]], args) -> None:
    name = os.path.basename(path)
    print(f"\n=== {name} ===")
    model = onnx.load(path)

    if args.simplify:
        print("- simplifying with fixed input shapes")
        model = simplify_model(model, input_shapes)

    name_gen = NameGen(model)
    if args.rewrite_layernorm:
        count = rewrite_layernorm(model, name_gen)
        if count:
            print(f"- expanded LayerNormalization: {count}")
    if args.rewrite_erf:
        count = rewrite_erf(model, name_gen)
        if count:
            print(f"- approximated Erf: {count}")
    if args.rewrite_sin:
        count = rewrite_sin(model, name_gen)
        if count:
            print(f"- rewrote Sin -> Cos: {count}")
    if args.rewrite_prelu:
        count = rewrite_prelu(model, name_gen)
        if count:
            print(f"- rewrote PRelu: {count}")
    if args.external_embed:
        if externalize_text_embedding(model, args.max_text_len):
            print("- externalized text embedding")
    if args.rewrite_pad:
        count = rewrite_pad(model)
        if count:
            print(f"- downgraded Pad inputs: {count}")
    if args.rewrite_axes:
        count = rewrite_axes_input_to_attr(model, "Unsqueeze")
        if count:
            print(f"- rewrote Unsqueeze axes: {count}")
        count = rewrite_axes_input_to_attr(model, "Squeeze")
        if count:
            print(f"- rewrote Squeeze axes: {count}")
        for op_type in ("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2"):
            count = rewrite_axes_input_to_attr(model, op_type)
            if count:
                print(f"- rewrote {op_type} axes: {count}")
    if args.rewrite_split:
        count = rewrite_split(model)
        if count:
            print(f"- rewrote Split inputs: {count}")
    if args.rewrite_clip:
        count = rewrite_clip_inputs(model)
        if count:
            print(f"- rewrote Clip inputs: {count}")

    if args.opset:
        set_opset(model, args.opset)

    if args.rewrite_clip_int:
        count = rewrite_clip_int_to_float(model, name_gen)
        if count:
            print(f"- casted int Clip ops: {count}")

    if args.clear_value_info:
        clear_value_info(model)
        print("- cleared value_info")

    if args.infer_shapes:
        model = infer_shapes(model)
        print("- inferred shapes")

    onnx_shapes = get_onnx_tensor_shapes(model)
    if args.rewrite_reshape:
        count = rewrite_reshape_dynamic(model, onnx_shapes)
        if count:
            print(f"- resolved Reshape -1 dims: {count}")

    base_name = os.path.splitext(name)[0]
    fixed_base = base_name if base_name.endswith("_fixed") else f"{base_name}_fixed"
    fixed_path = os.path.join(out_dir, f"{fixed_base}.onnx")
    onnx.save(model, fixed_path)
    print(f"- wrote {fixed_path}")

    supported = get_supported_ops()
    if supported:
        unsupported = list_unsupported_ops(model, supported)
        if unsupported:
            print(f"- unsupported ops remaining: {unsupported}")

    if args.dry_run:
        return

    patch_coreml_expand_dims()
    ensure_nnssa_stub()
    patch_onnx_coreml_dynamic_inputs()
    from onnx_coreml import convert  # type: ignore

    mlmodel = convert(fixed_path, minimum_ios_deployment_target=str(args.min_ios))
    out_path = os.path.join(out_dir, f"{os.path.splitext(name)[0]}.mlmodel")
    mlmodel.save(out_path)
    patch_prelu_input_tensors(out_path, onnx_shapes)
    print(f"- saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Supertonic ONNX to CoreML using onnx-coreml")
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--cfg-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-text-len", type=int, default=300)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--min-ios", type=int, default=13)
    parser.add_argument("--rewrite-layernorm", action="store_true")
    parser.add_argument("--rewrite-erf", action="store_true")
    parser.add_argument("--rewrite-sin", action="store_true")
    parser.add_argument("--rewrite-prelu", action="store_true")
    parser.add_argument("--rewrite-pad", action="store_true")
    parser.add_argument("--rewrite-axes", action="store_true")
    parser.add_argument("--rewrite-split", action="store_true")
    parser.add_argument("--rewrite-clip", action="store_true")
    parser.add_argument("--rewrite-clip-int", action="store_true")
    parser.add_argument("--rewrite-reshape", action="store_true")
    parser.add_argument("--external-embed", action="store_true")
    parser.add_argument("--clear-value-info", action="store_true")
    parser.add_argument("--infer-shapes", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg_dir = args.cfg_dir or args.onnx_dir
    cfgs = load_cfgs(cfg_dir)
    shapes = build_input_shapes(cfgs, args.max_text_len, args.max_seconds)

    models = [
        "duration_predictor.onnx",
        "text_encoder.onnx",
        "vector_estimator.onnx",
        "vocoder.onnx",
    ]

    for name in models:
        path = os.path.join(args.onnx_dir, name)
        if not os.path.exists(path):
            fixed_name = f"{os.path.splitext(name)[0]}_fixed.onnx"
            fixed_path = os.path.join(args.onnx_dir, fixed_name)
            if os.path.exists(fixed_path):
                path = fixed_path
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        convert_one(path, args.out_dir, shapes[name], args)


if __name__ == "__main__":
    main()
