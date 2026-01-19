"""
Database path configuration using BALDER_DB_PATH environment variable.

Usage:
    from tools.db_config import get_default_db_path

    parser.add_argument('--db-path', default=get_default_db_path())

Setup:
    export BALDER_DB_PATH="/full/path/to/balder-harmonized-biomarker-data-v20250522.sqlite"
"""

import os

def get_default_db_path():
    """
    Get database path from BALDER_DB_PATH env var or use default.

    Returns default path relative to balder_analysis parent directory.
    """
    return os.getenv(
        'BALDER_DB_PATH',
        '../../../output/actionability_db_curration_20250522/balder-harmonized-biomarker-data-v20250522.sqlite'
    )
