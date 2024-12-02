import logging

# Define the logger 
logger = logging.getLogger('analyzerportfolio')
logger.setLevel(logging.INFO)  # Default level

# Add a default console handler and ensure there are no duplicate handlers
if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)