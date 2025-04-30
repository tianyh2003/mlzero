import torch
import torch.nn as nn
import onnx
import numpy as np

from network_wrapper import NetworkWrapper

if __name__ == "__main__":
    plane_size = 3
    board_size = 8
    action_size = 64
    logger = None
    ckpt_dir = "./ckpt"
    use_gpu = False
    is_train = False

    network_wrapper = NetworkWrapper(
        plane_size=plane_size,
        board_size=board_size,
        action_size=action_size,
        logger=logger,
        ckpt_dir=ckpt_dir,
        use_gpu=use_gpu,
        is_train=is_train
    )
    network_wrapper.eval_mode()

    dummy_input = torch.randn(1, plane_size, board_size, board_size)
    torch.onnx.export(
        network_wrapper.nn,
        dummy_input,
        "alphazero.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    onnx_model = onnx.load("alphazero.onnx")
    onnx.checker.check_model(onnx_model)
    print("Model is valid ONNX model.")

