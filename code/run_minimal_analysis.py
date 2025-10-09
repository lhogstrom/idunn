#!/usr/bin/env python3
"""
Minimal OncoKB Analysis Entry Point

Simple script to run basic end-to-end analysis.
Usage: python run_minimal_analysis.py
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add the code directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from oncokb_analyzer.config import Config
from oncokb_analyzer.analyzer import OncoKBAnalyzer


def main():
    """Run minimal OncoKB analysis with default configuration"""

    print("OncoKB Minimal Analysis")
    print("=" * 30)

    try:
        # Use default configuration
        config = Config.default()
        print(f"Using database: {config.db_path}")
        print(f"Cancer types: {config.cancer_types}")
        print(f"Algorithm: {config.algorithm}")
        print(f"CV Strategy: {config.cv_strategy}")
        if config.cv_strategy == "group":
            print(f"Group Column: {config.group_column}")
        print()

        # Initialize analyzer
        analyzer = OncoKBAnalyzer(config)

        # Run analysis
        results = analyzer.run_analysis()

        # Display summary
        print("\nAnalysis Summary:")
        print("-" * 20)
        print(analyzer.get_summary(results))

        # Save results
        output_path = Path(config.output_dir) / "minimal_analysis_results.json"
        output_path.parent.mkdir(exist_ok=True)

        # Convert results to JSON-serializable format
        def convert_to_serializable(obj):
            """Convert numpy types to Python types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        json_results = convert_to_serializable(results)

        # Remove model objects which can't be serialized
        for cancer_type, result in json_results.get('cancer_type_results', {}).items():
            if 'model' in result:
                del result['model']

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"ERROR: Database file not found. {e}")
        print("Please check the database path in the configuration.")
        return 1

    except Exception as e:
        print(f"ERROR: Analysis failed: {e}")
        return 1

    print("\nAnalysis completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())