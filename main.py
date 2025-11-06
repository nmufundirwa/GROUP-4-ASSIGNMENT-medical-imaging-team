from pathlib import Path
import argparse
from src.pipelines.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Detection Pipeline")
    parser.add_argument("--input", required=True, help="Input directory containing images")
    parser.add_argument("--output", required=True, help="Output directory for processed images")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of images to process")
    args = parser.parse_args()
    run_pipeline(Path(args.input), Path(args.output), n=args.limit)
