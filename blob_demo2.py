import time
from datetime import timedelta

import blobconverter
import cv2
import depthai as dai
import numpy as np

# Custom JET colormap with 0 mapped to `black` - better disparity visualization
jet_custom = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
jet_custom[0] = [0, 0, 0]

blob = dai.OpenVINO.Blob("E:/Desktop/GSoC/boat/models/DDRNet/BEST_360.blob")
for name, tensorInfo in blob.networkInputs.items():
    print(name, tensorInfo.dims)
INPUT_SHAPE = blob.networkInputs["0"].dims[:2]
TARGET_SHAPE = (400, 400)


def decode_deeplabv3p(output_tensor):
    class_colors = [[0, 0, 0], [0, 255, 0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    output = output_tensor.reshape(*INPUT_SHAPE)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors


def get_multiplier(output_tensor):
    class_binary = [[0], [1]]
    class_binary = np.asarray(class_binary, dtype=np.uint8)
    output = output_tensor.reshape(*INPUT_SHAPE)
    output_colors = np.take(class_binary, output, axis=0)
    return output_colors


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


class HostSync:
    def __init__(self):
        self.arrays = {}

    def add_msg(self, name, msg):
        if not name in self.arrays:
            self.arrays[name] = []
        # Add msg to array
        self.arrays[name].append({"msg": msg})
        # Try finding synced msgs
        ts = msg.getTimestamp()
        synced = {}
        for name, arr in self.arrays.items():
            for i, obj in enumerate(arr):
                time_diff = abs(obj["msg"].getTimestamp() - ts)
                # 20ms since we add rgb/depth frames at 30FPS => 33ms. If
                # time difference is below 20ms, it's considered as synced
                if time_diff < timedelta(milliseconds=33):
                    synced[name] = obj["msg"]
                    # print(f"{name}: {i}/{len(arr)}")
                    break
        # If there are 3 (all) synced msgs, remove all old msgs
        # and return synced msgs
        if len(synced) == 3:  # color, depth, nn

            def remove(t1, t2):
                return timedelta(milliseconds=500) < abs(t1 - t2)

            # Remove old msgs
            for name, arr in self.arrays.items():
                for i, obj in enumerate(arr):
                    if remove(obj["msg"].getTimestamp(), ts):
                        arr.remove(obj)
                    else:
                        break
            return synced
        return False


def crop_to_square(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    delta = int((width - height) / 2)
    # print(height, width, delta)
    return frame[0:height, delta : width - delta]


# Start defining a pipeline
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# Color cam: 1920x1080
# Mono cam: 640x400
cam.setIspScale((1, 3), (1, 3))  # To match 400P mono cameras
cam.setBoardSocket(dai.CameraBoardSocket.RGB)

# For deeplabv3
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setPreviewSize(*INPUT_SHAPE)
cam.setInterleaved(False)

# NN output linked to XLinkOut
isp_xout = pipeline.create(dai.node.XLinkOut)
isp_xout.setStreamName("cam")
cam.isp.link(isp_xout.input)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setBlob(blob)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)
cam.preview.link(detection_nn.input)

# NN output linked to XLinkOut
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Left mono camera
left = pipeline.create(dai.node.MonoCamera)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
# Right mono camera
right = pipeline.create(dai.node.MonoCamera)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
left.out.link(stereo.left)
right.out.link(stereo.right)

# Create depth output
xout_disp = pipeline.create(dai.node.XLinkOut)
xout_disp.setStreamName("disparity")
stereo.disparity.link(xout_disp.input)

# Pipeline is defined, now we can connect to the device
with dai.Device() as device:
    cams = device.getConnectedCameras()
    depth_enabled = (
        dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
    )
    if not depth_enabled:
        raise RuntimeError(
            "Unable to run this experiment on device without depth capabilities! (Available cameras: {})".format(
                cams
            )
        )
    device.startPipeline(pipeline)
    # Output queues will be used to get the outputs from the device
    q_color = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    q_disp = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    fps = FPSHandler()
    sync = HostSync()
    disp_frame = None
    disp_multiplier = 255 / stereo.initialConfig.getMaxDisparity()

    frame = None
    depth = None
    depth_weighted = None
    frames = {}

    while True:
        msgs = q_nn.get()

        fps.next_iter()
        # get layer1 data
        layer1 = msgs.getFirstLayerFp16()
        # reshape to numpy array
        frame = np.asarray(layer1).reshape(360, 640) > 0.5
        frame = frame.astype(np.uint8) * 255

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
