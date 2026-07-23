from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Optional

import cv2


class CameraCaptureError(RuntimeError):
    pass


class CameraCaptureService:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def default_backend() -> int:
        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            return cv2.CAP_AVFOUNDATION
        return cv2.CAP_ANY

    def capture_image(
        self,
        device_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        warmup_frames: int = 10,
        filename: Optional[str] = None,
        backend: Optional[int] = None,
    ) -> Path:
        if warmup_frames < 0:
            raise ValueError("warmup_frames must be greater than or equal to 0")

        output_path = self.output_dir / self.build_filename(filename)
        camera, resolved_backend = self.open_camera(
            device_index=device_index,
            backend=backend,
        )

        try:
            if width is not None:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height is not None:
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            success, frame = self.read_frame(
                camera=camera,
                warmup_frames=warmup_frames,
            )

            if not success or frame is None:
                raise CameraCaptureError(
                    f"Camera device {device_index} did not return a frame. "
                    f"Backend={resolved_backend}."
                )

            if not cv2.imwrite(str(output_path), frame):
                raise CameraCaptureError(
                    f"Failed to write captured image to {output_path}."
                )

            return output_path
        finally:
            camera.release()

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

    def capture_image_from_candidates(
        self,
        device_indices: list[int],
        width: Optional[int] = None,
        height: Optional[int] = None,
        warmup_frames: int = 10,
        filename: Optional[str] = None,
        backend: Optional[int] = None,
    ) -> Path:
        if not device_indices:
            raise ValueError("device_indices must not be empty")

        errors: list[str] = []
        for device_index in device_indices:
            try:
                return self.capture_image(
                    device_index=device_index,
                    width=width,
                    height=height,
                    warmup_frames=warmup_frames,
                    filename=filename,
                    backend=backend,
                )
            except CameraCaptureError as error:
                errors.append(f"{device_index}: {error}")

        raise CameraCaptureError(
            "Could not capture from any requested camera device. "
            + " | ".join(errors)
        )

    def probe_devices(
        self,
        device_indices: list[int],
        width: Optional[int] = None,
        height: Optional[int] = None,
        warmup_frames: int = 5,
    ) -> list[dict[str, str | int | bool]]:
        results: list[dict[str, str | int | bool]] = []

        for device_index in device_indices:
            probe_filename = f"probe_camera_{device_index}.jpg"
            try:
                output_path = self.capture_image(
                    device_index=device_index,
                    width=width,
                    height=height,
                    warmup_frames=warmup_frames,
                    filename=probe_filename,
                )
                results.append(
                    {
                        "device_index": device_index,
                        "ok": True,
                        "image_path": str(output_path),
                    }
                )
            except (CameraCaptureError, ValueError) as error:
                results.append(
                    {
                        "device_index": device_index,
                        "ok": False,
                        "error": str(error),
                    }
                )

        return results

    @staticmethod
    def build_filename(filename: Optional[str]) -> str:
        if filename:
            candidate = Path(filename).name
            if not candidate:
                raise ValueError("filename must not be empty")
            if Path(candidate).suffix:
                return candidate
            return f"{candidate}.jpg"

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"capture_{timestamp}.jpg"
