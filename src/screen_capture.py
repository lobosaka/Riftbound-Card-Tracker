from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Optional


class ScreenCaptureError(RuntimeError):
    pass


class ScreenCaptureService:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_screen(self, filename: Optional[str] = None) -> Path:
        output_path = self.output_dir / self.build_filename(filename)
        result = subprocess.run(
            ["screencapture", "-x", str(output_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise ScreenCaptureError(
                "Screen capture failed. "
                f"stderr: {result.stderr.strip() or 'no error output'}"
            )

        if not output_path.exists() or not output_path.is_file():
            raise ScreenCaptureError(
                f"Screen capture finished but no image was written to {output_path}."
            )

        return output_path

    @staticmethod
    def build_filename(filename: Optional[str]) -> str:
        if filename:
            candidate = Path(filename).name
            if not candidate:
                raise ValueError("filename must not be empty")
            if Path(candidate).suffix:
                return candidate
            return f"{candidate}.png"

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"screen_{timestamp}.png"
