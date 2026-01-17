#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import torch
import coremltools as ct
from onnx import TensorProto, defs
from onnx2torch import convert
from onnx2torch.node_converters import pad as pad_mod
from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription
from onnx2torch.utils.common import OperationConverterResult, OnnxMapping

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import convert_onnx_coreml as conv  # noqa: E402


MODEL_FILES = [
    "duration_predictor.onnx",
    "text_encoder.onnx",
    "vector_estimator.onnx",
    "vocoder.onnx",
]

OUTPUT_NAMES = {
    "duration_predictor": "duration",
    "text_encoder": "text_emb",
    "vector_estimator": "denoised_latent",
    "vocoder": "wav_tts",
}


def patch_onnx2torch_pad() -> None:
    def _pad_converter_static_when_possible(node, graph):
        onnx_mode = node.attributes.get("mode", "constant")
        mode = pad_mod._onnx_to_torch_mode(onnx_mode)
        pads_name = node.input_values[1] if len(node.input_values) > 1 else None
        value_name = node.input_values[2] if len(node.input_values) > 2 else None
        if pads_name and pads_name in graph.initializers:
            pads_arr = graph.initializers[pads_name].to_numpy().flatten().tolist()
            constant_value = 0.0
            if value_name and value_name in graph.initializers:
                val_arr = graph.initializers[value_name].to_numpy().flatten()
                if val_arr.size:
                    constant_value = float(val_arr[0])
            torch_module = pad_mod.OnnxPadStatic.create_from_onnx_params(
                onnx_pads=pads_arr,
                onnx_mode=onnx_mode,
                constant_value=constant_value,
            )
            return OperationConverterResult(
                torch_module=torch_module,
                onnx_mapping=OnnxMapping(inputs=(node.input_values[0],), outputs=node.output_values),
            )

        return OperationConverterResult(
            torch_module=pad_mod.OnnxPadDynamic(mode=mode),
            onnx_mapping=OnnxMapping(inputs=node.input_values, outputs=node.output_values),
        )

    _CONVERTER_REGISTRY[
        OperationDescription(domain=defs.ONNX_DOMAIN, operation_type="Pad", version=11)
    ] = _pad_converter_static_when_possible


def get_input_infos(model: onnx.ModelProto) -> List[Tuple[str, List[int], int]]:
    init_names = {i.name for i in model.graph.initializer}
    infos: List[Tuple[str, List[int], int]] = []
    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        t = inp.type.tensor_type
        if not t.HasField("shape"):
            raise ValueError(f"Missing input shape for {inp.name}")
        dims: List[int] = []
        for d in t.shape.dim:
            if d.dim_value > 0:
                dims.append(int(d.dim_value))
            else:
                raise ValueError(f"Non-static input dim for {inp.name}")
        infos.append((inp.name, dims, t.elem_type))
    return infos


def make_example_inputs(infos: List[Tuple[str, List[int], int]]) -> Tuple[Tuple[torch.Tensor, ...], List[ct.TensorType]]:
    tensors: List[torch.Tensor] = []
    ct_inputs: List[ct.TensorType] = []
    for name, shape, elem_type in infos:
        if elem_type in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE):
            tensor = torch.randn(*shape, dtype=torch.float32)
        elif elem_type in (TensorProto.INT32, TensorProto.INT64):
            tensor = torch.zeros(*shape, dtype=torch.int64)
        else:
            raise ValueError(f"Unsupported input dtype for {name}: {elem_type}")
        tensors.append(tensor)
        ct_inputs.append(ct.TensorType(name=name, shape=shape))
    return tuple(tensors), ct_inputs


def mlprogram_target(min_ios: int):
    if min_ios < 15:
        raise ValueError("ML Program requires iOS 15 or newer")
    if min_ios >= 18 and hasattr(ct.target, "iOS18"):
        return ct.target.iOS18
    if min_ios >= 17 and hasattr(ct.target, "iOS17"):
        return ct.target.iOS17
    if min_ios >= 16 and hasattr(ct.target, "iOS16"):
        return ct.target.iOS16
    return ct.target.iOS15


def apply_rewrites(model: onnx.ModelProto, args, input_shapes: Dict[str, List[int]]) -> onnx.ModelProto:
    if args.simplify:
        print("- simplifying with fixed input shapes")
        model = conv.simplify_model(model, input_shapes)

    name_gen = conv.NameGen(model)
    if args.rewrite_layernorm:
        count = conv.rewrite_layernorm(model, name_gen)
        if count:
            print(f"- expanded LayerNormalization: {count}")
    if args.rewrite_erf:
        count = conv.rewrite_erf(model, name_gen)
        if count:
            print(f"- approximated Erf: {count}")
    if args.rewrite_sin:
        count = conv.rewrite_sin(model, name_gen)
        if count:
            print(f"- rewrote Sin -> Cos: {count}")
    if args.rewrite_prelu:
        count = conv.rewrite_prelu(model, name_gen)
        if count:
            print(f"- rewrote PRelu: {count}")
    if args.external_embed:
        if conv.externalize_text_embedding(model, args.max_text_len):
            print("- externalized text embedding")
    if args.rewrite_pad:
        count = conv.rewrite_pad(model)
        if count:
            print(f"- downgraded Pad inputs: {count}")
    if args.rewrite_axes:
        count = conv.rewrite_axes_input_to_attr(model, "Unsqueeze")
        if count:
            print(f"- rewrote Unsqueeze axes: {count}")
        count = conv.rewrite_axes_input_to_attr(model, "Squeeze")
        if count:
            print(f"- rewrote Squeeze axes: {count}")
        for op_type in ("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2"):
            count = conv.rewrite_axes_input_to_attr(model, op_type)
            if count:
                print(f"- rewrote {op_type} axes: {count}")
    if args.rewrite_split:
        count = conv.rewrite_split(model)
        if count:
            print(f"- rewrote Split inputs: {count}")
    if args.rewrite_clip:
        count = conv.rewrite_clip_inputs(model)
        if count:
            print(f"- rewrote Clip inputs: {count}")

    if args.opset:
        conv.set_opset(model, args.opset)

    if args.rewrite_clip_int:
        count = conv.rewrite_clip_int_to_float(model, name_gen)
        if count:
            print(f"- casted int Clip ops: {count}")

    if args.clear_value_info:
        conv.clear_value_info(model)
        print("- cleared value_info")

    if args.infer_shapes:
        model = conv.infer_shapes(model)
        print("- inferred shapes")

    if args.rewrite_reshape:
        onnx_shapes = conv.get_onnx_tensor_shapes(model)
        count = conv.rewrite_reshape_dynamic(model, onnx_shapes)
        if count:
            print(f"- resolved Reshape -1 dims: {count}")

    return model


def save_fixed(model: onnx.ModelProto, out_dir: str, name: str) -> str:
    base_name = os.path.splitext(name)[0]
    fixed_base = base_name if base_name.endswith("_fixed") else f"{base_name}_fixed"
    fixed_path = os.path.join(out_dir, f"{fixed_base}.onnx")
    onnx.save(model, fixed_path)
    print(f"- wrote {fixed_path}")
    return fixed_path


def rename_output(model_path: str, new_name: str) -> None:
    spec = ct.models.utils.load_spec(model_path)
    if len(spec.description.output) != 1:
        print(f"- skip rename for {model_path} (unexpected output count)")
        return
    old_name = spec.description.output[0].name
    if old_name == new_name:
        return
    ct.models.utils.rename_feature(spec, old_name, new_name, rename_inputs=False, rename_outputs=True)
    if model_path.endswith(".mlpackage"):
        model_spec_path = os.path.join(model_path, "Data", "com.apple.CoreML", "model.mlmodel")
        ct.models.utils.save_spec(spec, model_spec_path)
    else:
        ct.models.utils.save_spec(spec, model_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Supertonic-2 ONNX to CoreML ML Program.")
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--cfg-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fixed-onnx-dir", required=True)
    parser.add_argument("--max-text-len", type=int, default=300)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--min-ios", type=int, default=15)
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
    parser.add_argument("--rename-outputs", action="store_true")
    args = parser.parse_args()

    os.makedirs(".coremltmp", exist_ok=True)
    os.environ["TMPDIR"] = os.path.abspath(".coremltmp")

    os.makedirs(args.fixed_onnx_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg_dir = args.cfg_dir or args.onnx_dir
    cfgs = conv.load_cfgs(cfg_dir)
    shapes = conv.build_input_shapes(cfgs, args.max_text_len, args.max_seconds)

    patch_onnx2torch_pad()
    target = mlprogram_target(args.min_ios)

    for name in MODEL_FILES:
        in_path = os.path.join(args.onnx_dir, name)
        if not os.path.exists(in_path):
            raise FileNotFoundError(in_path)
        print(f"\n=== {name} ===")
        model = onnx.load(in_path)
        model = apply_rewrites(model, args, shapes[name])
        fixed_path = save_fixed(model, args.fixed_onnx_dir, name)

        print("- converting with onnx2torch + coremltools")
        torch_model = convert(fixed_path)
        torch_model.eval()

        infos = get_input_infos(model)
        example_inputs, ct_inputs = make_example_inputs(infos)

        traced = torch.jit.trace(torch_model, example_inputs, strict=False)
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=ct_inputs,
            minimum_deployment_target=target,
        )

        base = os.path.splitext(name)[0]
        out_path = os.path.join(args.out_dir, f"{base}_mlprogram.mlpackage")
        mlmodel.save(out_path)
        print(f"- saved {out_path}")

        if args.rename_outputs:
            new_name = OUTPUT_NAMES.get(base)
            if new_name:
                rename_output(out_path, new_name)
                print(f"- renamed output to {new_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
