from pathlib import Path
from ..image_loader import list_images, read_image, preprocess_image
from ..image_processor import apply_gaussian_blur, apply_clahe, save_image
from ..classifier import classify_image

def run_pipeline(input_dir: Path, output_dir: Path, n: int = 10):
    """
    Run detection pipeline on chest x-ray images.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        n: Maximum number of images to process
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create subdirectories for Normal and Pneumonia images
    normal_dir = output_dir / "Normal"
    pneumonia_dir = output_dir / "Pneumonia"
    normal_dir.mkdir(parents=True, exist_ok=True)
    pneumonia_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    print(f"Found {len(images)} images in {input_dir}")
    images = images[:n]
    
    for p in images:
        try:
            # Classify the image
            classification = classify_image(p)
            print(f"Image {p.name} classified as {classification}")

            # Process the image
            img = read_image(p)
            img = preprocess_image(img, target_size=(224, 224))
            enhanced = apply_clahe(img)
            blurred = apply_gaussian_blur(enhanced, ksize=5)

            # Save the image to the appropriate directory
            if classification == "Normal":
                out_path = normal_dir / p.name
            else:
                out_path = pneumonia_dir / p.name
            
            save_image(out_path, blurred)
            
        except Exception as e:
            print(f"Error processing {p.name}: {e}")
            continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Image Detection Pipeline")
    parser.add_argument("--input", required=True, help="Input directory containing images")
    parser.add_argument("--output", required=True, help="Output directory for processed images")
    args = parser.parse_args()
    run_pipeline(Path(args.input), Path(args.output), n=20)