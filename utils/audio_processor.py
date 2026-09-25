import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
import yt_dlp
import imageio_ffmpeg

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
FFMPEG_BIN_DIR = os.path.join(tempfile.gettempdir(), "clipmind-ffmpeg")
os.makedirs(FFMPEG_BIN_DIR, exist_ok=True)
FFMPEG_COMMAND = os.path.join(FFMPEG_BIN_DIR, "ffmpeg")
if os.path.islink(FFMPEG_COMMAND) and os.path.realpath(FFMPEG_COMMAND) != os.path.realpath(FFMPEG_PATH):
    os.unlink(FFMPEG_COMMAND)
if not os.path.exists(FFMPEG_COMMAND):
    os.symlink(FFMPEG_PATH, FFMPEG_COMMAND)
os.environ["PATH"] = FFMPEG_BIN_DIR + os.pathsep + os.environ.get("PATH", "")

warnings.filterwarnings(
    "ignore",
    message="Couldn't find ffprobe or avprobe.*",
    category=RuntimeWarning,
    module="pydub.utils",
)
from pydub import AudioSegment

AudioSegment.converter = FFMPEG_PATH

def download_youtube_audio(url: str, work_dir: str) -> str:
    output_path = os.path.join(work_dir, "recording.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "ffmpeg_location": FFMPEG_PATH,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)
    wav_files = list(Path(work_dir).glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError("YouTube audio download did not produce a WAV file.")
    return str(wav_files[0])



def convert_to_wav(input_path: str, work_dir: str) -> str:
    """Convert audio or video to mono 16 kHz WAV."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = os.path.join(work_dir, "converted.wav")
    result = subprocess.run(
        [
            FFMPEG_PATH,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            input_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        details = result.stderr.strip()[-600:] or "Unsupported or unreadable audio/video format."
        raise RuntimeError(f"Audio conversion failed: {details}")
    return output_path



def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks

def process_input(source: str) -> list:
    work_dir = tempfile.mkdtemp(prefix="clipmind-audio-")
    try:
        if source.startswith("http://") or source.startswith("https://"):
            print("Detected YouTube URL. Downloading audio...")
            wav_path = download_youtube_audio(source, work_dir)
        else:
            print("Detected local file. Converting to WAV...")
            wav_path = convert_to_wav(source, work_dir)

        print("Chunking audio...")
        chunks = chunk_audio(wav_path)
        print(f"Audio ready — {len(chunks)} chunk(s) created.")
        chunk_paths = {os.path.abspath(path) for path in chunks}
        for item in Path(work_dir).iterdir():
            if item.is_file() and str(item.resolve()) not in chunk_paths:
                item.unlink(missing_ok=True)
        return chunks
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def cleanup_input_chunks(chunks: list) -> None:
    """Remove chunk files and their per-analysis temporary directory."""
    work_dirs = set()
    for chunk_path in chunks:
        path = Path(chunk_path)
        work_dirs.add(path.parent)
        path.unlink(missing_ok=True)

    for work_dir in work_dirs:
        if work_dir.name.startswith("clipmind-audio-"):
            shutil.rmtree(work_dir, ignore_errors=True)
