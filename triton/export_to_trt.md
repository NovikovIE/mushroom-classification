1. sudo docker run -it --rm --gpus '"device=0"' --runtime=nvidia -v
   ./export:/models nvcr.io/nvidia/tensorrt:24.07-py3

2. trtexec --onnx=/models/onnx/model.onnx
   --saveEngine=/models/tensorrt/model.plan --minShapes=input:1x3x224x224
   --optShapes=input:16x3x224x224 --maxShapes=input:64x3x224x224
   --profilingVerbosity=detailed --builderOptimizationLevel=5 --fp16

3. cp export/tensorrt/model.plan model_repository/trt/1/model.plan
