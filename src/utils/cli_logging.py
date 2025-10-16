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


def configure_cli_logging(verbose: int = 1, log_file: str = None):
    """
    Configure logging for CLI with appropriate verbosity.

    Args:
        verbose: Verbosity level (0=silent, 1=normal, 2=detailed, 3=debug)
        log_file: Optional file path to save detailed logs

    Example:
        >>> configure_cli_logging(verbose=2, log_file="analysis.log")
        >>> # Now all loggers will respect these settings
    """
    # Map verbosity levels to logging levels
    if verbose == 0:
        console_level = logging.ERROR  # Only errors
    elif verbose == 1:
        console_level = logging.WARNING  # Normal output - warnings and errors
    elif verbose == 2:
        console_level = logging.INFO  # Detailed output - info, warnings, errors
    elif verbose == 3:
        console_level = logging.DEBUG  # Debug output - everything
    else:  # verbose >= 4
        console_level = logging.CRITICAL  # Summary-only - silence all logs, only show table
    file_level = logging.DEBUG  # Always detailed in file

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything
    root_logger.handlers.clear()  # Remove existing handlers

    # Console handler - clean output
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)

    if verbose >= 4:
        # Summary-only mode - no console logging (remove console handler)
        # Don't add console handler to root logger
        pass
    elif verbose >= 3:
        # Verbose debugging with timestamp and logger name
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
    elif verbose >= 2:
        # Detailed output with minimal info
        console_formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        console_handler.setFormatter(console_formatter)
    else:
        # Minimal format for warnings/errors only
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)

    # Only add console handler if not in summary-only mode
    if verbose < 4:
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

    # Suppress noisy third-party loggers based on verbosity
    if verbose <= 1:
        # Normal mode - suppress most third-party noise
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        logging.getLogger('charset_normalizer').setLevel(logging.WARNING)
        logging.getLogger('yfinance').setLevel(logging.WARNING)
    elif verbose == 2:
        # Detailed mode - show some third-party info
        logging.getLogger('httpx').setLevel(logging.INFO)
        logging.getLogger('httpcore').setLevel(logging.INFO)
        logging.getLogger('urllib3').setLevel(logging.INFO)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        logging.getLogger('charset_normalizer').setLevel(logging.WARNING)
        logging.getLogger('yfinance').setLevel(logging.INFO)
    elif verbose == 3:
        # Debug mode - show all third-party info
        logging.getLogger('httpx').setLevel(logging.DEBUG)
        logging.getLogger('httpcore').setLevel(logging.DEBUG)
        logging.getLogger('urllib3').setLevel(logging.DEBUG)
        logging.getLogger('asyncio').setLevel(logging.DEBUG)
        logging.getLogger('charset_normalizer').setLevel(logging.DEBUG)
        logging.getLogger('yfinance').setLevel(logging.DEBUG)

    # Our own internal loggers - set appropriate levels
    if verbose <= 1:
        # Normal mode - hide implementation details
        logging.getLogger('src.data.cache').setLevel(logging.WARNING)
        logging.getLogger('src.llm.cache').setLevel(logging.WARNING)
        logging.getLogger('src.communication.batch_manager').setLevel(logging.WARNING)
        logging.getLogger('src.data.pipeline').setLevel(logging.WARNING)
    elif verbose == 2:
        # Detailed mode - show some internal info
        logging.getLogger('src.data.cache').setLevel(logging.INFO)
        logging.getLogger('src.llm.cache').setLevel(logging.INFO)
        logging.getLogger('src.communication.batch_manager').setLevel(logging.INFO)
        logging.getLogger('src.data.pipeline').setLevel(logging.INFO)
    elif verbose == 3:
        # Debug mode - show all internal details
        logging.getLogger('src.data.cache').setLevel(logging.DEBUG)
        logging.getLogger('src.llm.cache').setLevel(logging.DEBUG)
        logging.getLogger('src.communication.batch_manager').setLevel(logging.DEBUG)
        logging.getLogger('src.data.pipeline').setLevel(logging.DEBUG)

    # Configure structlog if available
    if HAS_STRUCTLOG:
        # Configure structlog for CLI usage
        if verbose == 0:
            structlog_level = logging.ERROR
        elif verbose == 1:
            structlog_level = logging.WARNING
        elif verbose == 2:
            structlog_level = logging.INFO
        elif verbose == 3:
            structlog_level = logging.DEBUG
        else:  # verbose >= 4
            structlog_level = logging.CRITICAL

        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]

        if verbose >= 4:
            # Summary-only mode - no logging output
            processors.append(structlog.processors.KeyValueRenderer(
                key_order=['level', 'event'],
                drop_missing=True
            ))
        elif verbose >= 3:
            # Detailed format for debug mode
            processors.append(structlog.dev.ConsoleRenderer())
        elif verbose >= 2:
            # Moderate format for detailed mode
            processors.append(structlog.processors.KeyValueRenderer(
                key_order=['event', 'level'],
                drop_missing=True
            ))
        else:
            # Minimal format for normal/silent mode
            processors.append(structlog.processors.KeyValueRenderer(
                key_order=['level', 'event'],
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
