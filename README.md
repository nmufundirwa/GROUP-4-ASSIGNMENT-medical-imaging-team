# Medical Image Processor and Classifier

This project is a Python-based medical imaging processor that provides a set of tools for loading, processing, classifying, and saving medical images.

## Project Structure

```
├── .gitignore
├── README.md
├── images
│   ├── ...
├── main.py
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── classifier.py
│   ├── image_loader.py
│   ├── image_processor.py
│   └── pipelines
│       └── pipeline.py
└── tests
    └── test_pipeline.py
```

## How to Run

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**

   ```bash
   python main.py --input images --output output
   ```

   This will process chest x-ray images and save them into `Normal` and `Pneumonia` subdirectories based on the classification.

## How to Run Tests

To run the tests, use the following command:

```bash
python -m unittest discover tests
```
