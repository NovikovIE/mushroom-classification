import io

import numpy as np
import torchvision.transforms as transforms
import triton_python_backend_utils as pb_utils
from PIL import Image


class TritonPythonModel:
    def initialize(self, args):
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input_image")
            raw_image_bytes = input_tensor.as_numpy().tobytes()

            image = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB")
            tensor = self.transform(image).numpy().astype(np.float32)

            out_tensor = np.expand_dims(tensor, axis=0)

            output_tensor = pb_utils.Tensor("processed_tensor", out_tensor)

            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses
