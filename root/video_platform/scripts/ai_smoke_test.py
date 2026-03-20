import importlib
import os
import shutil
import subprocess
import sys


def check_command(name):
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"missing command: {name}")
    print(f"{name}: {path}")


def run(cmd):
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def main():
    print(f"python: {sys.version.split()[0]}")
    print(f"base_dir: {os.environ.get('VIDEO_PLATFORM_BASE_DIR', '/root/video_platform')}")

    check_command("ffmpeg")
    check_command("ffprobe")
    check_command("tesseract")

    langs = run(["tesseract", "--list-langs"])
    if "chi_sim" not in langs or "eng" not in langs:
        raise RuntimeError("tesseract languages missing chi_sim or eng")
    print("tesseract languages: ok")

    for module_name in [
        "cv2",
        "dashscope",
        "numpy",
        "PIL",
        "pytesseract",
        "requests",
        "torch",
        "whisper",
        "core.correct_asr",
        "core.generate_questions",
        "core.generate_web_title_summary",
        "core.knowledge_graph",
        "core.relation_network",
        "core.video_summary",
    ]:
        importlib.import_module(module_name)
        print(f"import ok: {module_name}")

    import torch

    cuda_available = torch.cuda.is_available()
    print(f"torch cuda available: {cuda_available}")
    if cuda_available:
        print(f"torch cuda version: {torch.version.cuda}")
        print(f"torch cuda device count: {torch.cuda.device_count()}")

    expect_cuda = os.environ.get("EXPECT_CUDA")
    if expect_cuda and expect_cuda.strip().lower() in {"1", "true", "yes", "on"} and not cuda_available:
        raise RuntimeError("EXPECT_CUDA is set but torch.cuda.is_available() is False")

    print("ai smoke test: success")


if __name__ == "__main__":
    main()
