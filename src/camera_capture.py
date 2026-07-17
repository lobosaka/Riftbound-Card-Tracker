from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2


class CameraCaptureError(RuntimeError):
    pass


class CameraCaptureService:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_image(
        self,
        device_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        warmup_frames: int = 10,
        filename: Optional[str] = None,
    ) -> Path:
        if warmup_frames < 0:
            raise ValueError("warmup_frames must be greater than or equal to 0")

        output_path = self.output_dir / self.build_filename(filename)
        camera = cv2.VideoCapture(device_index)

        if not camera.isOpened():
            raise CameraCaptureError(
                f"Could not open camera device {device_index}. "
                "Make sure the camera is connected and available as a webcam/UVC device."
            )

        try:
            if width is not None:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height is not None:
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            success = False
            frame = None

            for _ in range(max(1, warmup_frames)):
                success, frame = camera.read()

            if not success or frame is None:
                raise CameraCaptureError(
                    f"Camera device {device_index} did not return a frame."
                )

            if not cv2.imwrite(str(output_path), frame):
                raise CameraCaptureError(
                    f"Failed to write captured image to {output_path}."
                )

            return output_path
        finally:
            camera.release()

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
