import time

import cv2
import depthai as dai
import numpy as np

blob = dai.OpenVINO.Blob(
    "./models/WithDepth(DDRNet)/640_360_U8.blob"
)  # TODO MODEL PATH
for name, tensorInfo in blob.networkInputs.items():
    print(name, tensorInfo.dims)
INPUT_SHAPE = blob.networkInputs["rgb"].dims[:2]


class FPSHandler:
    def __init__(self):
        self.timestamp = time.time()
        self.start = time.time()
        self.frame_cnt = 0

    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)


# Start defining a pipeline
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setIspScale((1, 3), (1, 3))  # TODO RGB->640x360
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
cam.setPreviewSize(*INPUT_SHAPE)
cam.setInterleaved(False)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setBlob(blob)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)
cam.preview.link(detection_nn.inputs["rgb"])

# Left mono camera
left = pipeline.create(dai.node.MonoCamera)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
# Right mono camera
right = pipeline.create(dai.node.MonoCamera)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
left.out.link(stereo.left)
right.out.link(stereo.right)

# Depth output linked to NN
stereo.depth.link(detection_nn.inputs["depth"])

# NN output linked to XLinkOut
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)
xout_img = pipeline.create(dai.node.XLinkOut)
xout_img.setStreamName("img")
detection_nn.passthroughs["rgb"].link(xout_img.input)
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
detection_nn.passthroughs["depth"].link(xout_depth.input)


# Pipeline is defined, now we can connect to the device
with dai.Device() as device:
    device.startPipeline(pipeline)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)  # type: ignore
    q_img = device.getOutputQueue(name="img", maxSize=4, blocking=False)  # type: ignore
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)  # type: ignore
    fps = FPSHandler()
    while True:
        msgs = q_nn.get()
        img = q_img.get().getCvFrame()
        depth = q_depth.get().getFrame()
        fps.next_iter()

        # get layer1 data
        layer1 = msgs.getLayerFp16("out")
        # reshape to numpy array
        distance_vector = np.asarray(layer1)
        print(distance_vector)
        print(distance_vector.shape)

        depth = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(
            np.uint8
        )
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        frame = np.concatenate((img, depth), axis=0)

        cv2.putText(
            frame,
            "Fps: {:.2f}".format(fps.fps()),
            (2, frame.shape[0] - 4),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.4,
            color=(255, 255, 255),
        )
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break
