import sys
from pathlib import Path

import depthai as dai
import numpy as np

"""# Bootloader Info
(res, info) = dai.DeviceBootloader.getFirstAvailableDevice()
if res == True:
    print(f"Found device with name: {info.name}")
    bl = dai.DeviceBootloader(info)
    print(f"Version: {bl.getVersion()}")

    supportedMemTypes = [
        dai.DeviceBootloader.Memory.FLASH,
        dai.DeviceBootloader.Memory.EMMC,
    ]
    if bl.getType() == dai.DeviceBootloader.Type.USB:
        print("USB Bootloader - supports only Flash memory")
        supportedMemTypes = [dai.DeviceBootloader.Memory.FLASH]
    else:
        print("NETWORK Bootloader")

    try:
        for mem in supportedMemTypes:
            memInfo = bl.getMemoryInfo(mem)
            if memInfo.available:
                print(f"Memory '{mem}' size: {memInfo.size}, info: {memInfo.info}")
                appInfo = bl.readApplicationInfo(mem)
                if appInfo.hasApplication:
                    print(
                        f"Application name: {appInfo.applicationName}, firmware version: {appInfo.firmwareVersion}"
                    )
            else:
                print(f"Memory '{mem.name}' not available...")
    except Exception as ex:
        print(f"Couldn't retrieve memory details: {ex}")
else:
    print("No devices found")"""

pipeline = dai.Pipeline()
with dai.Device(pipeline) as device:
    device.setLogLevel(dai.LogLevel.DEBUG)
    device.setLogOutputLevel(dai.LogLevel.DEBUG)

    # Print Myriad X Id (MxID), USB speed, and available cameras on the device
    print("MxId:", device.getDeviceInfo().getMxId())
    print("USB speed:", device.getUsbSpeed())
    print("Connected cameras:", device.getConnectedCameras())

    # Calibration Info
    calibFile = str(
        (Path(__file__).parent / Path(f"calib_{device.getMxId()}.json"))
        .resolve()
        .absolute()
    )
    if len(sys.argv) > 1:
        calibFile = sys.argv[1]

    calibData = device.readCalibration()
    calibData.eepromToJsonFile(calibFile)

    M_rgb, width, height = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.RGB)
    print("RGB Camera Default intrinsics...")
    print(M_rgb)
    print(width)
    print(height)

    if (
        "OAK-1" in calibData.getEepromData().boardName
        or "BW1093OAK" in calibData.getEepromData().boardName
    ):
        M_rgb = np.array(
            calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, 640, 360)
        )
        print("RGB Camera resized intrinsics...")
        print(M_rgb)

        D_rgb = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RGB))
        print("RGB Distortion Coefficients...")
        [
            print(name + ": " + value)
            for (name, value) in zip(
                [
                    "k1",
                    "k2",
                    "p1",
                    "p2",
                    "k3",
                    "k4",
                    "k5",
                    "k6",
                    "s1",
                    "s2",
                    "s3",
                    "s4",
                    "τx",
                    "τy",
                ],
                [str(data) for data in D_rgb],
            )
        ]

        print(f"RGB FOV {calibData.getFov(dai.CameraBoardSocket.RGB)}")

    else:
        M_left = np.array(
            calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, 640, 360)
        )
        print("LEFT Camera resized intrinsics...")
        print(M_left)

        D_left = np.array(
            calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT)
        )
        print("LEFT Distortion Coefficients...")
        [
            print(name + ": " + value)
            for (name, value) in zip(
                [
                    "k1",
                    "k2",
                    "p1",
                    "p2",
                    "k3",
                    "k4",
                    "k5",
                    "k6",
                    "s1",
                    "s2",
                    "s3",
                    "s4",
                    "τx",
                    "τy",
                ],
                [str(data) for data in D_left],
            )
        ]

        M_right = np.array(
            calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 640, 360)
        )
        print("RIGHT Camera resized intrinsics...")
        print(M_right)

        D_right = np.array(
            calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT)
        )
        print("RIGHT Distortion Coefficients...")
        [
            print(name + ": " + value)
            for (name, value) in zip(
                [
                    "k1",
                    "k2",
                    "p1",
                    "p2",
                    "k3",
                    "k4",
                    "k5",
                    "k6",
                    "s1",
                    "s2",
                    "s3",
                    "s4",
                    "τx",
                    "τy",
                ],
                [str(data) for data in D_right],
            )
        ]

        print(
            f"RGB FOV {calibData.getFov(dai.CameraBoardSocket.RGB)}, Mono FOV {calibData.getFov(dai.CameraBoardSocket.LEFT)}"
        )

        R1 = np.array(calibData.getStereoLeftRectificationRotation())
        R2 = np.array(calibData.getStereoRightRectificationRotation())
        M_right = np.array(
            calibData.getCameraIntrinsics(calibData.getStereoRightCameraId(), 640, 360)
        )

        H_left = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_left))
        print("LEFT Camera stereo rectification matrix...")
        print(H_left)

        H_right = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_right))
        print("RIGHT Camera stereo rectification matrix...")
        print(H_right)

        lr_extrinsics = np.array(
            calibData.getCameraExtrinsics(
                dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT
            )
        )
        print(
            "Transformation matrix of where left Camera is W.R.T right Camera's optical center"
        )
        print(lr_extrinsics)

        l_rgb_extrinsics = np.array(
            calibData.getCameraExtrinsics(
                dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RGB
            )
        )
        print(
            "Transformation matrix of where left Camera is W.R.T RGB Camera's optical center"
        )
        print(l_rgb_extrinsics)
