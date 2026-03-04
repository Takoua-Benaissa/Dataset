"""
Configuration for the Text-to-Image Dataset Creation Pipeline.
All tunable parameters are centralized here.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ImageQualityConfig:
    """Strict image quality thresholds."""
    min_resolution: int = 256          # Minimum width AND height in pixels
    min_sharpness: float = 200.0       # Laplacian variance threshold
    min_brightness: float = 30.0       # Minimum mean pixel intensity
    max_brightness: float = 225.0      # Maximum mean pixel intensity
    min_contrast: float = 30.0         # Minimum standard deviation of pixels


@dataclass
class OCRConfig:
    """OCR extraction parameters."""
    min_confidence: int = 75           # Minimum OCR confidence (%)
    min_text_length: int = 1           # Minimum extracted text length
    max_text_length: int = 50          # Maximum extracted text length
    tesseract_lang: str = "eng"        # Tesseract language(s)
    tesseract_psm: int = 11            # Page segmentation mode (11 = sparse text)
    require_letter: bool = True        # Must contain at least one letter [A-Za-z]


@dataclass
class BLIPConfig:
    """BLIP captioning model parameters."""
    model_name: str = "Salesforce/blip-image-captioning-large"
    max_length: int = 75               # Maximum caption token length
    num_beams: int = 5                 # Beam search width
    temperature: float = 0.7           # Generation temperature


@dataclass
class OutputConfig:
    """Output dataset specifications."""
    target_size: int = 256             # Output image size (square)
    jpeg_quality: int = 95             # JPEG compression quality
    min_images: int = 100              # Minimum acceptable dataset size
    max_images: int = 300              # Stop after reaching this count
    train_ratio: float = 0.80          # Train split ratio
    val_ratio: float = 0.10            # Validation split ratio
    test_ratio: float = 0.10           # Test split ratio


@dataclass
class DownloadConfig:
    """Data download parameters."""
    textcaps_train_url: str = (
        "https://dl.fbaipublicfiles.com/textvqa/data/textcaps/"
        "TextCaps_0.1_train.json"
    )
    textcaps_val_url: str = (
        "https://dl.fbaipublicfiles.com/textvqa/data/textcaps/"
        "TextCaps_0.1_val.json"
    )
    # TextVQA train images (shared with TextCaps)
    textvqa_images_url: str = (
        "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"
    )
    download_timeout: int = 30         # Per-image download timeout (seconds)
    max_download_workers: int = 8      # Parallel download threads
    max_download_retries: int = 2      # Retry failed downloads


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    image_quality: ImageQualityConfig = field(default_factory=ImageQualityConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    blip: BLIPConfig = field(default_factory=BLIPConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    batch_size: int = 16               # Processing batch size
    seed: int = 42                     # Random seed for reproducibility
    log_every: int = 50                # Log progress every N images

    @property
    def prompt_template(self) -> str:
        return "{blip_caption}, with the text '{ocr_text}' clearly visible"
