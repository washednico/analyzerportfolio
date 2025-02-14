import logging
import plotly.io as pio
from .logger import logger
from typing import Optional

## -- PLOTLY CONFIGURATION -- ##

# Default settings
DEFAULT_TEMPLATE = "plotly_dark"
CUSTOM_TEMPLATES = {
    "transparent": {
        "layout": {
            "plot_bgcolor": "rgba(0,0,0,0)",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "font": {"color": "black"},
        }
    }
}

# Register custom templates
for template_name, template_config in CUSTOM_TEMPLATES.items():
    pio.templates[template_name] = template_config

# Current settings
_plotly_template = DEFAULT_TEMPLATE
_use_transparent = False


def set_plotly_template(template: str = DEFAULT_TEMPLATE, transparent: bool = False) -> None:
    """
    Set the global Plotly template and transparency option.

    Parameters
    ----------
    template : str, optional
        The name of the Plotly template to set as default. Default is "plotly".
    transparent : bool, optional
        Whether to use the transparent template. Default is False.
    """
    global _plotly_template, _use_transparent
    _use_transparent = transparent

    if transparent:
        template = "transparent"

    if template not in pio.templates:
        logger.warning(f"Invalid template '{template}'. Falling back to '{DEFAULT_TEMPLATE}'.")
        template = DEFAULT_TEMPLATE
    else:
        logger.info(f"Using custom template '{template}'.")

    _plotly_template = template
    pio.templates.default = template
    logger.info(f"Set Plotly template to '{template}' with transparent={transparent}.")


def get_plotly_template() -> str:
    """
    Retrieve the current global Plotly template.

    Returns
    -------
    str
        The name of the current global Plotly template.
    """
    return _plotly_template


def is_transparent() -> bool:
    """
    Check if the transparent template is currently in use.

    Returns
    -------
    bool
        True if the transparent template is active, otherwise False.
    """
    return _use_transparent




## -- LOGGING CONFIGURATION -- ##


def reset_logging():
    """
    Resets the logging configuration by clearing all handlers.
    """
    if logger.hasHandlers():
        logger.handlers.clear()

def get_logger() -> logging.Logger:
    """
    Retrieve the global logger for advanced configuration.

    Returns
    -------
    logging.Logger
        The global package logger.
    """
    return logger

def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_level: Optional[int] = None,
    verbose: bool = False,
    style: str = "detailed",
) -> None:
    """
    Configures logging for the package.

    Parameters
    ----------
    level : int
        Overall logging level for the package logger.
    log_file : str, optional
        File path to save logs. If None, logs are only displayed in the console.
    console_level : int, optional
        Specific logging level for the console handler. Defaults to the global level.
    verbose : bool, optional
        If True, console logs display at DEBUG level.
    style : str, optional
        Logging style. Options are:
        - "detailed": Includes timestamps, levels, and logger names.
        - "print_like": Logs appear simple, like print statements.
    """
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print_like_formatter = logging.Formatter('%(message)s')
    formatter = detailed_formatter if style == "detailed" else print_like_formatter

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG if verbose else (console_level or level))
    logger.addHandler(console_handler)

    # File handler (if log_file is specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
