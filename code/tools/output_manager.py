"""
Minimalist output path management for Balder project
Creates timestamped directories with consistent structure
"""

import os
from pathlib import Path
from datetime import datetime


class OutputManager:
    """Manage output directories with consistent naming"""

    def __init__(self, output_root=None):
        """
        Initialize output manager

        Args:
            output_root: Base output directory (overrides env var)
        """
        if output_root is None:
            # Try environment variable first
            output_root = os.getenv('BALDER_OUTPUT_ROOT')

        if output_root is None:
            # Fallback to ../output/ relative to repo
            output_root = Path(__file__).parent.parent.parent / 'output'

        self.root = Path(output_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_output_dir(self, category, name, use_timestamp=True):
        """
        Create output directory with optional timestamp

        Args:
            category: Subdirectory category (e.g., 'ml_experiments', 'analysis/demographics')
            name: Descriptive name (e.g., 'feature_comparison')
            use_timestamp: Include timestamp in directory name (default: True)

        Returns:
            Path object for the created directory

        Example:
            >>> output_mgr = OutputManager()
            >>> output_dir = output_mgr.create_output_dir('ml_experiments', 'feature_comparison')
            >>> # Returns: /path/to/output/ml_experiments/feature_comparison_20251106_143022/
        """
        # Create directory name
        if use_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dir_name = f"{name}_{timestamp}"
        else:
            dir_name = name

        # Create full path
        output_dir = self.root / category / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {output_dir}")

        return output_dir


# Simple usage example
if __name__ == "__main__":
    # Test the output manager
    output_mgr = OutputManager()

    test_dir = output_mgr.create_output_dir(
        category='ml_experiments',
        name='test_run',
        use_timestamp=True
    )

    print(f"Created: {test_dir}")
