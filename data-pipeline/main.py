"""
CLI orchestrator for TacticalGraph data pipeline.

Accepts --mode full|incremental and --dev / --dev-comp flags to drive the
complete ETL sequence: schema install → reference → entities → transfers →
matches → appearances → game events, with watermark-based incremental support.
"""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Optional

from config import get_settings
from database import Neo4jDatabase
from dataset import DatasetManager
from schema import SchemaInstaller
from watermark import WatermarkManager
from loaders.reference_loader import ReferenceLoader
from loaders.entities_loader import EntitiesLoader
from loaders.transfers_loader import TransfersLoader
from loaders.matches_loader import MatchesLoader
from loaders.appearances_loader import AppearancesLoader
from loaders.game_events_loader import GameEventsLoader

# region Logging setup
LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(verbose: bool = False) -> None:
    """
    Configure root logger with structured formatting.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT, stream=sys.stdout)
    # Suppress noisy third-party loggers
    for noisy in ("neo4j", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger(__name__)
# endregion


# region Argument parsing
def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="tacticalgraph-pipeline",
        description="TacticalGraph ETL pipeline — loads Kaggle transfermarkt data into Neo4j.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help=(
            "Ingestion mode:\n"
            "  full        — Load all CSV data, overwriting existing nodes/edges.\n"
            "  incremental — Load only games with date > last watermark date.\n"
            "(default: full)"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        metavar="PATH",
        help="Optional custom path to directory containing raw CSV files (default: dynamically downloaded/cached via kagglehub).",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Enable DEV_MODE: filters pipeline to Premier League ('GB1') subgraph.",
    )
    parser.add_argument(
        "--dev-comp",
        default="GB1",
        metavar="COMP_ID",
        help=(
            "Target competition ID for DEV_MODE filtering "
            "(default: 'GB1' — Premier League). Implies --dev."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level log output.",
    )
    return parser
# endregion


# region Pipeline steps
def _timed_step(label: str, fn, *args, **kwargs):
    """
    Execute a pipeline step, logging elapsed time and any errors.

    Args:
        label: Human-readable name for the step.
        fn: Callable to execute.
        *args, **kwargs: Forwarded to fn.

    Returns:
        The return value of fn, or re-raises on exception.
    """
    logger.info("━━━ STEP START: %s ━━━", label)
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info("━━━ STEP DONE:  %s  (%.2fs) ━━━", label, elapsed)
        return result
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error("━━━ STEP FAILED: %s  (%.2fs) — %s ━━━", label, elapsed, e)
        raise
# endregion


# region Main entrypoint
def main() -> None:
    """
    Main CLI entrypoint for the TacticalGraph ETL pipeline.

    Exit codes:
        0 — Pipeline completed successfully.
        1 — Pipeline aborted due to an unrecoverable error.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    # --dev-comp implies --dev
    dev_mode: bool = args.dev or (args.dev_comp != "GB1")
    dev_comp: str = args.dev_comp
    mode: str = args.mode

    pipeline_start = time.perf_counter()
    settings = get_settings()

    # Step 0: Resolve dataset path (via DatasetManager or local CLI override)
    if args.data_dir:
        dataset_path = Path(args.data_dir)
        logger.info("Using explicit local data directory: %s", dataset_path)
    else:
        dataset_manager = DatasetManager(settings=settings)
        dataset_path = _timed_step("Acquire Kaggle Dataset", dataset_manager.get_dataset_path)

    logger.info("=" * 60)
    logger.info("  TacticalGraph Data Pipeline")
    logger.info("  mode=%s | dev_mode=%s | dev_comp=%s | dataset_path=%s",
                mode, dev_mode, dev_comp, dataset_path)
    logger.info("=" * 60)

    # Shared database connection
    db = Neo4jDatabase(settings=settings)

    try:
        db.connect()

        # Step 1: Schema constraints + indexes
        _timed_step(
            "Schema Installation",
            SchemaInstaller(db=db).install_schema,
        )

        # Step 2: Watermark (incremental mode)
        watermark_manager = WatermarkManager(db=db)
        watermark_date: Optional[str] = None

        if mode == "incremental":
            watermark_date = _timed_step(
                "Read Watermark",
                watermark_manager.get_last_processed_date,
            )
            if watermark_date:
                logger.info("Incremental mode: loading games after %s", watermark_date)
            else:
                logger.warning(
                    "No watermark found; incremental run will load all available data."
                )

        # Step 3: Reference data
        _timed_step(
            "Reference Loader (countries, competitions, national teams)",
            ReferenceLoader(db=db, settings=settings).load,
            dataset_path,
            dev_mode,
        )

        # Step 4: Entity data
        _timed_step(
            "Entities Loader (clubs, players, valuations)",
            EntitiesLoader(db=db, settings=settings).load,
            dataset_path,
            dev_mode,
        )

        # Step 5: Transfer data
        _timed_step(
            "Transfers Loader",
            TransfersLoader(db=db, settings=settings).load,
            dataset_path,
            dev_mode,
        )

        # Step 6: Match data
        _timed_step(
            "Matches Loader (games, club_games)",
            MatchesLoader(db=db, settings=settings).load,
            dataset_path,
            dev_mode,
            watermark_date,  # passes through for incremental date filter
        )

        # Step 7: Appearance data
        _timed_step(
            "Appearances Loader (appearances + game_lineups)",
            AppearancesLoader(db=db, settings=settings).load,
            dataset_path,
            dev_mode,
        )

        # Step 8: Game event data
        _timed_step(
            "Game Events Loader",
            GameEventsLoader(db=db, settings=settings).load,
            dataset_path,
            dev_mode,
        )

        # Step 9: Update watermark
        import pandas as pd

        games_path = dataset_path / "games.csv"
        if games_path.exists():
            try:
                games_df = pd.read_csv(games_path, usecols=["date"])
                games_df["date"] = pd.to_datetime(games_df["date"], errors="coerce")
                latest_date = games_df["date"].dropna().max()
                if pd.notna(latest_date):
                    latest_date_str = latest_date.strftime("%Y-%m-%d")
                    _timed_step(
                        f"Update Watermark → {latest_date_str}",
                        watermark_manager.update_last_processed_date,
                        latest_date_str,
                    )
            except Exception as wm_err:
                logger.warning("Could not determine latest game date for watermark: %s", wm_err)

        total_elapsed = time.perf_counter() - pipeline_start
        logger.info("=" * 60)
        logger.info("  Pipeline completed successfully in %.2fs", total_elapsed)
        logger.info("=" * 60)
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (KeyboardInterrupt).")
        sys.exit(1)
    except Exception as exc:
        total_elapsed = time.perf_counter() - pipeline_start
        logger.critical(
            "Pipeline aborted after %.2fs with unrecoverable error: %s",
            total_elapsed,
            exc,
            exc_info=True,
        )
        sys.exit(1)
    finally:
        db.close()
# endregion


if __name__ == "__main__":
    main()