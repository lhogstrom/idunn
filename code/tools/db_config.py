"""Database path configuration via the BALDER_DB_PATH environment variable.

Usage: from tools.db_config import get_default_db_path
"""

import os

def get_default_db_path():
    """Return the database path from BALDER_DB_PATH, or a default relative path."""
    return os.getenv(
        'BALDER_DB_PATH',
        '../../../output/actionability_db_curration_20250522/balder-harmonized-biomarker-data-v20250522.sqlite'
    )
