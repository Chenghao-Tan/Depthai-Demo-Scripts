import time

import cv2
import depthai as dai
import numpy as np

GRID_NUM_H = 10  # TODO
GRID_NUM_W = 10  # TODO

blob = dai.OpenVINO.Blob(
    "./models/DDRNet(WithDepth)/640_360_(10_10)_debug.blob"
)  # TODO MODEL PATH (USE DEBUG VERSION)
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
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
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
        grids = msgs.getLayerFp16("out")
        filtered = msgs.getLayerFp16("debug")
        img = q_img.get().getCvFrame()
        depth = q_depth.get().getFrame()
        fps.next_iter()

        grids = np.asarray(grids).reshape(GRID_NUM_H, GRID_NUM_W, 4)  # lxyz
        filtered = np.asarray(filtered).reshape(INPUT_SHAPE[1], INPUT_SHAPE[0])

        filtered = (
            (filtered - filtered.min()) / (filtered.max() - filtered.min()) * 255
        ).astype(np.uint8)
        filtered = cv2.resize(filtered, (1280, 720))  # Force 720P for bigger display

        for i in range(1, GRID_NUM_H):
            cv2.line(
                filtered,
                (0, int(filtered.shape[0] * i / GRID_NUM_H)),
                (filtered.shape[1] - 1, int(filtered.shape[0] * i / GRID_NUM_H)),
                color=(255, 255, 255),
                thickness=1,
            )
        for i in range(1, GRID_NUM_W):
            cv2.line(
                filtered,
                (int(filtered.shape[1] * i / GRID_NUM_W), 0),
                (int(filtered.shape[1] * i / GRID_NUM_W), filtered.shape[0] - 1),
                color=(255, 255, 255),
                thickness=1,
            )

        for i in range(GRID_NUM_H):
            for j in range(GRID_NUM_W):
                cv2.putText(
                    filtered,
                    "label:{:d}".format(grids[i][j][0].astype(np.uint8)),
                    (
                        int(filtered.shape[1] * j / GRID_NUM_W) + 3,
                        int(filtered.shape[0] * i / GRID_NUM_H) + 12,
                    ),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.4,
                    color=(255, 255, 255),
                )
                cv2.putText(
                    filtered,
                    "x:{:.1f}m".format(grids[i][j][1].astype(np.uint8)),
                    (
                        int(filtered.shape[1] * j / GRID_NUM_W) + 3,
                        int(filtered.shape[0] * i / GRID_NUM_H) + 24,
                    ),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.4,
                    color=(255, 255, 255),
                )
                cv2.putText(
                    filtered,
                    "y:{:.1f}m".format(grids[i][j][2].astype(np.uint8)),
                    (
                        int(filtered.shape[1] * j / GRID_NUM_W) + 3,
                        int(filtered.shape[0] * i / GRID_NUM_H) + 36,
                    ),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.4,
                    color=(255, 255, 255),
                )
                cv2.putText(
                    filtered,
                    "z:{:.1f}m".format(grids[i][j][3].astype(np.uint8)),
                    (
                        int(filtered.shape[1] * j / GRID_NUM_W) + 3,
                        int(filtered.shape[0] * i / GRID_NUM_H) + 48,
                    ),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.4,
                    color=(255, 255, 255),
                )

        cv2.putText(
            filtered,
            "Fps: {:.2f}".format(fps.fps()),
            (2, filtered.shape[0] - 4),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.4,
            color=(255, 255, 255),
        )
        cv2.imshow("FILTERED", filtered)
        cv2.imshow("RGB", img)
        depth = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(
            np.uint8
        )
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        cv2.imshow("DEPTH", depth)

        if cv2.waitKey(1) == ord("q"):
            break
