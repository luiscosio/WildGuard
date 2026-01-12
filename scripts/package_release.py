"""Package models and data for release/download."""

import argparse
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RELEASE_DIR = PROJECT_ROOT / "release"


def create_models_zip(output_path: Path = None) -> Path:
    """Create zip of trained models."""
    if output_path is None:
        output_path = RELEASE_DIR / f"DarkPatternMonitor_models_{datetime.now().strftime('%Y%m%d')}.zip"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    classifier_dir = MODELS_DIR / "classifier"
    if not classifier_dir.exists():
        print(f"Warning: {classifier_dir} does not exist")
        return None

    print(f"Creating models archive: {output_path}")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add classifier model files
        for file_path in classifier_dir.glob("*"):
            if file_path.is_file():
                arcname = f"classifier/{file_path.name}"
                print(f"  Adding: {arcname}")
                zf.write(file_path, arcname)

    print(f"Models archive created: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return output_path


def create_data_zip(output_path: Path = None, version: str = "v5") -> Path:
    """Create zip of detection outputs and reports."""
    if output_path is None:
        output_path = RELEASE_DIR / f"DarkPatternMonitor_data_{datetime.now().strftime('%Y%m%d')}.zip"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    version_dir = OUTPUTS_DIR / version
    if not version_dir.exists():
        print(f"Warning: {version_dir} does not exist")
        return None

    print(f"Creating data archive: {output_path}")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add version-specific outputs
        for file_path in version_dir.glob("*"):
            if file_path.is_file() and file_path.suffix in ['.json', '.jsonl']:
                arcname = f"{version}/{file_path.name}"
                print(f"  Adding: {arcname}")
                zf.write(file_path, arcname)

        # Add root-level reports if they exist
        for report_file in ['prevalence.json', 'gap_report.json', 'reliability_report.json',
                           'topic_analysis.json', 'analytics.json']:
            report_path = OUTPUTS_DIR / report_file
            if report_path.exists():
                print(f"  Adding: {report_file}")
                zf.write(report_path, report_file)

    print(f"Data archive created: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return output_path


def create_training_data_zip(output_path: Path = None) -> Path:
    """Create zip of labeled training data."""
    if output_path is None:
        output_path = RELEASE_DIR / f"DarkPatternMonitor_training_{datetime.now().strftime('%Y%m%d')}.zip"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    labeled_dir = DATA_DIR / "labeled"
    if not labeled_dir.exists():
        print(f"Warning: {labeled_dir} does not exist")
        return None

    print(f"Creating training data archive: {output_path}")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in labeled_dir.glob("*.jsonl"):
            arcname = f"labeled/{file_path.name}"
            print(f"  Adding: {arcname}")
            zf.write(file_path, arcname)

    print(f"Training data archive created: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Package DarkPatternMonitor release files")
    parser.add_argument("--models", action="store_true", help="Create models zip")
    parser.add_argument("--data", action="store_true", help="Create data zip")
    parser.add_argument("--training", action="store_true", help="Create training data zip")
    parser.add_argument("--all", action="store_true", help="Create all zips")
    parser.add_argument("--version", default="v5", help="Data version to package (default: v5)")
    parser.add_argument("--output-dir", type=str, help="Output directory for zips")

    args = parser.parse_args()

    if args.output_dir:
        global RELEASE_DIR
        RELEASE_DIR = Path(args.output_dir)

    if args.all or (not args.models and not args.data and not args.training):
        # Default: create all
        create_models_zip()
        create_data_zip(version=args.version)
        create_training_data_zip()
    else:
        if args.models:
            create_models_zip()
        if args.data:
            create_data_zip(version=args.version)
        if args.training:
            create_training_data_zip()

    print(f"\nRelease files created in: {RELEASE_DIR}")


if __name__ == "__main__":
    main()
