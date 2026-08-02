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

# List of essential CSV files required by primary domain loaders
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
    "countries.csv",
    "national_teams.csv",
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

    def _download_with_retry(self, handle: Optional[str] = None) -> str:
        """
        Execute kagglehub.dataset_download with tenacity retry backoff.

        Args:
            handle: Optional Kaggle dataset handle to download. Defaults to self.handle.

        Returns:
            str: Path string returned by kagglehub.
        """
        target_handle = handle or self.handle

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
            logger.info("Requesting Kaggle dataset download/cache check for handle: '%s'", target_handle)
            path_str = kagglehub.dataset_download(target_handle)
            return path_str

        return _download()

    def get_dataset_path(
        self,
        handle: Optional[str] = None,
        required_files: Optional[List[str]] = None,
    ) -> Path:
        """
        Download or retrieve cached dataset path via kagglehub and verify required CSV files exist.

        Args:
            handle: Optional Kaggle dataset handle override. If None, uses self.handle.
            required_files: Optional list of required CSV filenames to verify. If None, uses REQUIRED_CSV_FILES.

        Returns:
            Path: Verified Path object pointing to the dataset directory containing raw CSV files.

        Raises:
            FileNotFoundError: If the downloaded directory or required CSV files are missing.
            RuntimeError: If download fails after maximum retry attempts.
        """
        target_handle = handle or self.handle
        check_files = required_files if required_files is not None else REQUIRED_CSV_FILES

        try:
            downloaded_str = self._download_with_retry(handle=target_handle)
            dataset_path = Path(downloaded_str)

            if not dataset_path.exists() or not dataset_path.is_dir():
                error_msg = f"Dataset path returned by kagglehub does not exist or is not a directory: {dataset_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            if check_files:
                logger.info("Dataset available at path: %s. Verifying required CSV files...", dataset_path)
                missing_files: List[str] = [
                    csv_file for csv_file in check_files if not (dataset_path / csv_file).is_file()
                ]

                if missing_files:
                    error_msg = (
                        f"Dataset directory '{dataset_path}' for handle '{target_handle}' is missing required CSV files: {missing_files}"
                    )
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)

                logger.info(
                    "Successfully verified all %d required CSV files at %s.",
                    len(check_files),
                    dataset_path,
                )
            else:
                logger.info("Successfully acquired dataset for handle '%s' at %s.", target_handle, dataset_path)

            return dataset_path

        except (FileNotFoundError, RuntimeError):
            raise
        except Exception as e:
            logger.error("Failed to acquire dataset for handle '%s': %s", target_handle, e, exc_info=True)
            raise RuntimeError(f"Failed to acquire Kaggle dataset '{target_handle}': {e}") from e
