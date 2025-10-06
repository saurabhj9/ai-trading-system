"""
Logging configuration for CLI with clean output.

This module provides centralized logging configuration for the CLI,
allowing clean output by default with verbose mode for debugging.
"""
import logging
import sys
from pathlib import Path

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


def configure_cli_logging(verbose: bool = False, log_file: str = None):
    """
    Configure logging for CLI with appropriate verbosity.

    Args:
        verbose: If True, show DEBUG messages. If False, show only WARNING+
        log_file: Optional file path to save detailed logs

    Example:
        >>> configure_cli_logging(verbose=True, log_file="analysis.log")
        >>> # Now all loggers will respect these settings
    """
    # Determine log level
    console_level = logging.DEBUG if verbose else logging.WARNING
    file_level = logging.DEBUG  # Always detailed in file

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything
    root_logger.handlers.clear()  # Remove existing handlers

    # Console handler - clean output
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)

    if verbose:
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        # Minimal format for warnings/errors only
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler - detailed logs (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('charset_normalizer').setLevel(logging.WARNING)

    # Our own internal loggers - set appropriate levels
    if not verbose:
        # These are implementation details users don't need to see
        logging.getLogger('src.data.cache').setLevel(logging.WARNING)
        logging.getLogger('src.llm.cache').setLevel(logging.WARNING)
        logging.getLogger('src.communication.batch_manager').setLevel(logging.WARNING)
        logging.getLogger('src.data.pipeline').setLevel(logging.WARNING)

    # Configure structlog if available
    if HAS_STRUCTLOG:
        # Configure structlog for CLI usage
        structlog_level = logging.DEBUG if verbose else logging.WARNING

        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]

        if verbose:
            # Detailed format for verbose mode
            processors.append(structlog.dev.ConsoleRenderer())
        else:
            # Minimal format for normal mode - only show level and message
            processors.append(structlog.processors.KeyValueRenderer(
                key_order=['event', 'level'],
                drop_missing=True
            ))

        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Also set logging level for structlog's underlying logger
        logging.getLogger().setLevel(structlog_level)
