"""
Dataset manager module for TacticalGraph data pipeline.

Handles fetching, caching, and verifying Kaggle datasets using kagglehub with exponential backoff retries.
"""

import logging
from pathlib import Path
from typing import List, Optional

import kagglehub
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from config import Settings, get_settings

logger = logging.getLogger(__name__)

# List of essential CSV files required by domain loaders
REQUIRED_CSV_FILES: List[str] = [
    "players.csv",
    "games.csv",
    "appearances.csv",
    "clubs.csv",
    "competitions.csv",
    "transfers.csv",
    "club_games.csv",
    "game_events.csv",
    "player_valuations.csv",
    "game_lineups.csv",
]


class DatasetManager:
    """
    Manages downloading and verifying Kaggle datasets for the TacticalGraph pipeline.
    """

    def __init__(
        self,
        handle: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize DatasetManager.

        Args:
            handle: Kaggle dataset handle (e.g. 'davidcariboo/player-scores'). If None, uses settings.
            settings: Settings instance. If None, default settings are loaded.
        """
        self.settings = settings or get_settings()
        self.handle = handle or self.settings.KAGGLE_DATASET_HANDLE

    def _download_with_retry(self) -> str:
        """
        Execute kagglehub.dataset_download with tenacity retry backoff.

        Returns:
            str: Path string returned by kagglehub.
        """
        retry_decorator = retry(
            stop=stop_after_attempt(self.settings.MAX_RETRY_ATTEMPTS),
            wait=wait_exponential(
                multiplier=1,
                min=self.settings.RETRY_MIN_WAIT,
                max=self.settings.RETRY_MAX_WAIT,
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

        @retry_decorator
        def _download() -> str:
            logger.info("Requesting Kaggle dataset download/cache check for handle: '%s'", self.handle)
            path_str = kagglehub.dataset_download(self.handle)
            return path_str

        return _download()

    def get_dataset_path(self) -> Path:
        """
        Download or retrieve cached dataset path via kagglehub and verify all required CSV files exist.

        Returns:
            Path: Verified Path object pointing to the dataset directory containing raw CSV files.

        Raises:
            FileNotFoundError: If the downloaded directory or required CSV files are missing.
            RuntimeError: If download fails after maximum retry attempts.
        """
        try:
            downloaded_str = self._download_with_retry()
            dataset_path = Path(downloaded_str)

            if not dataset_path.exists() or not dataset_path.is_dir():
                error_msg = f"Dataset path returned by kagglehub does not exist or is not a directory: {dataset_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            logger.info("Dataset available at path: %s. Verifying required CSV files...", dataset_path)

            missing_files: List[str] = [
                csv_file for csv_file in REQUIRED_CSV_FILES if not (dataset_path / csv_file).is_file()
            ]

            if missing_files:
                error_msg = (
                    f"Dataset directory '{dataset_path}' is missing required CSV files: {missing_files}"
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            logger.info(
                "Successfully verified all %d required CSV files at %s.",
                len(REQUIRED_CSV_FILES),
                dataset_path,
            )
            return dataset_path

        except (FileNotFoundError, RuntimeError):
            raise
        except Exception as e:
            logger.error("Failed to acquire dataset for handle '%s': %s", self.handle, e, exc_info=True)
            raise RuntimeError(f"Failed to acquire Kaggle dataset '{self.handle}': {e}") from e
