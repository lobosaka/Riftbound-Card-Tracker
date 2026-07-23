import sys
import time
from typing import Optional

import cv2


class CameraCaptureError(RuntimeError):
    pass


class CameraCaptureService:
    @staticmethod
    def default_backend() -> int:
        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            return cv2.CAP_AVFOUNDATION
        return cv2.CAP_ANY

    def open_camera(
        self,
        device_index: int = 0,
        backend: Optional[int] = None,
    ) -> tuple[cv2.VideoCapture, int]:
        resolved_backend = (
            self.default_backend() if backend is None else backend
        )
        camera = cv2.VideoCapture(device_index, resolved_backend)

        if not camera.isOpened():
            raise CameraCaptureError(
                f"Could not open camera device {device_index}. "
                f"Backend={resolved_backend}. "
                "Make sure the camera is connected and available as a webcam/UVC device."
            )

        return camera, resolved_backend

    def open_camera_from_candidates(
        self,
        device_indices: list[int],
        backend: Optional[int] = None,
    ) -> tuple[cv2.VideoCapture, int, int]:
        if not device_indices:
            raise ValueError("device_indices must not be empty")

        errors: list[str] = []
        for device_index in device_indices:
            try:
                camera, resolved_backend = self.open_camera(
                    device_index=device_index,
                    backend=backend,
                )
                return camera, resolved_backend, device_index
            except CameraCaptureError as error:
                errors.append(f"{device_index}: {error}")

        raise CameraCaptureError(
            "Could not open any requested camera device. "
            + " | ".join(errors)
        )

    @staticmethod
    def read_frame(
        camera: cv2.VideoCapture,
        warmup_frames: int = 0,
    ) -> tuple[bool, Optional[object]]:
        success = False
        frame = None

        for _ in range(max(1, warmup_frames)):
            success, frame = camera.read()

        return success, frame

    @staticmethod
    def encode_jpeg(frame, jpeg_quality: int = 85) -> bytes:
        success, encoded_frame = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        if not success:
            raise CameraCaptureError("Failed to encode camera frame as JPEG.")
        return encoded_frame.tobytes()

    def mjpeg_stream(
        self,
        device_index: int = 0,
        device_indices: Optional[list[int]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        warmup_frames: int = 10,
        backend: Optional[int] = None,
        fps: float = 10.0,
        jpeg_quality: int = 85,
    ):
        if device_indices is None:
            camera, resolved_backend = self.open_camera(
                device_index=device_index,
                backend=backend,
            )
            active_device_index = device_index
        else:
            camera, resolved_backend, active_device_index = self.open_camera_from_candidates(
                device_indices=device_indices,
                backend=backend,
            )

        if width is not None:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        try:
            self.read_frame(camera=camera, warmup_frames=warmup_frames)
            frame_interval_seconds = 1.0 / fps if fps > 0 else 0.0

            while True:
                success, frame = camera.read()
                if not success or frame is None:
                    raise CameraCaptureError(
                        f"Camera device {active_device_index} did not return a frame. "
                        f"Backend={resolved_backend}."
                    )

                frame_bytes = self.encode_jpeg(
                    frame=frame,
                    jpeg_quality=jpeg_quality,
                )
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )

                if frame_interval_seconds > 0:
                    time.sleep(frame_interval_seconds)
        finally:
            camera.release()
