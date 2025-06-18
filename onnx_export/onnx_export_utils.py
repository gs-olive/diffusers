import onnx_graphsurgeon as gs
from polygraphy.backend.onnx.loader import fold_constants
import onnx
import tempfile
import os
import torch
from abc import ABC, abstractmethod


class BaseONNXExporter(ABC):
    def __init__(self, device="cuda", optimization_dtype="fp16"):
        self.device = device
        self.optimization_dtype = optimization_dtype
        self.model = None

    @abstractmethod
    def load_model(self):
        """Load the specific model for this exporter"""
        pass

    @abstractmethod
    def get_input_names(self):
        """Return list of input names for ONNX export"""
        pass

    @abstractmethod
    def get_output_names(self):
        """Return list of output names for ONNX export"""
        pass

    @abstractmethod
    def get_dynamic_axes(self):
        """Return dynamic axes dictionary for ONNX export"""
        pass

    @abstractmethod
    def get_sample_input(self, batch_size, **kwargs):
        """Generate sample input for ONNX export"""
        pass

    def get_torch_dtype(self):
        """Get torch dtype based on optimization_dtype"""
        if self.optimization_dtype == "fp16":
            return torch.float16
        elif self.optimization_dtype == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

    def export(self, output_path, batch_size=1, **kwargs):
        assert output_path.endswith(".onnx")
        
        print(f"[I] Exporting {self.__class__.__name__} ONNX model: {output_path}")

        # Load model if not already loaded
        if self.model is None:
            self.model = self.load_model()
            self.model.eval()

        with torch.no_grad(), tempfile.TemporaryDirectory() as tmp_dir:
            inputs = self.get_sample_input(batch_size, **kwargs)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_model_path = os.path.join(tmp_dir, os.path.basename(output_path))
                
                torch.onnx.export(
                    self.model,
                    inputs,
                    tmp_model_path,
                    export_params=True,
                    do_constant_folding=True,
                    opset_version=17,
                    input_names=self.get_input_names(),
                    output_names=self.get_output_names(),
                    dynamic_axes=self.get_dynamic_axes(),
                    verbose=False,
                )

                print(f"Intermediate model exported to: {tmp_model_path}")
                print(f"Optimizing ONNX model ...")
                onnx_opt_graph = Optimizer.optimize(onnx.load(tmp_model_path))
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                onnx.save_model(
                    onnx_opt_graph,
                    output_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    convert_attribute=False,
                    location=os.path.basename(output_path)[:-len(".onnx")] + ".data",
                )
                print(f"Optimized {self.__class__.__name__} model exported to: {output_path}")


class Optimizer:
    def __init__(self, onnx_graph):
        self.graph = gs.import_onnx(onnx_graph)

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        return gs.export_onnx(self.graph) if return_onnx else self.graph

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=False)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    @staticmethod
    def optimize(onnx_graph):
        opt = Optimizer(onnx_graph)
        opt.cleanup()
        opt.fold_constants()
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        return onnx_opt_graph
