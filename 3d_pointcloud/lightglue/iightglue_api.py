from lightglue_utils import (
    LightGlueRunner,
    load,
    rgb_to_grayscale,
)

class LightGlueAPI:

    def __init__(self):

        self.runner = LightGlueRunner(
            extractor_path="superpoint.onnx",
            lightglue_path="superpoint_lightglue.onnx",
            env_id=0
        )

    def match(
        self,
        image0_path,
        image1_path
    ):

        image0, scales0 = load(
            image0_path,
            resize=512
        )

        image1, scales1 = load(
            image1_path,
            resize=512
        )

        image0 = rgb_to_grayscale(image0)
        image1 = rgb_to_grayscale(image1)

        m_kpts0, m_kpts1 = self.runner.run(
            image0,
            image1,
            scales0,
            scales1
        )
        
        return m_kpts0, m_kpts1
