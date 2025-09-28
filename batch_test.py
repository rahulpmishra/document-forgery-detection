#!/usr/bin/env python3
"""
Batch Document Forgery Detection Tool

A command-line tool for testing multiple images and generating comprehensive reports.
Usage: python batch_test.py <dataset_path> [options]
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from keras.models import load_model
import io
from datetime import datetime
import time

class BatchForgeryDetector:
    def __init__(self, model_path="./models/trained_model.h5"):
        """Initialize the batch detector with model path"""
        self.model_path = model_path
        self.model = None
        self.class_names = ["Forged", "Authentic"]
        self.results = []
        self.stats = {
            "total_images": 0,
            "forged_count": 0,
            "authentic_count": 0,
            "errors": 0,
            "processing_time": 0
        }
        
    def load_model(self):
        """Load the trained model"""
        try:
            print(f"Loading model from: {self.model_path}")
            self.model = load_model(self.model_path)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def convert_to_ela_image_memory(self, image, quality=90):
        """Convert image to ELA without saving to disk"""
        try:
            # Convert to RGB if not already
            original_image = image.convert("RGB")
            
            # Save to memory buffer with specified quality
            buffer = io.BytesIO()
            original_image.save(buffer, "JPEG", quality=quality)
            buffer.seek(0)
            
            # Reload from buffer
            resaved_image = Image.open(buffer)
            
            # Calculate pixel difference
            ela_image = ImageChops.difference(original_image, resaved_image)
            
            # Calculate scaling factors
            extrema = ela_image.getextrema()
            max_difference = max([pix[1] for pix in extrema])
            if max_difference == 0:
                max_difference = 1
            scale = 350.0 / max_difference
            
            # Enhance brightness
            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
            
            return ela_image
        except Exception as e:
            print(f"ELA processing error: {str(e)}")
            return None

    def prepare_image_for_prediction(self, image):
        """Prepare image for CNN prediction"""
        try:
            image_size = (128, 128)
            ela_image = self.convert_to_ela_image_memory(image, 90)
            if ela_image is None:
                return None
            return np.array(ela_image.resize(image_size)).flatten() / 255.0
        except Exception as e:
            print(f"Image preparation error: {str(e)}")
            return None

    def predict_single_image(self, image_path):
        """Predict single image and return results"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Prepare for prediction
            test_image = self.prepare_image_for_prediction(image)
            if test_image is None:
                return None
                
            test_image = test_image.reshape(-1, 128, 128, 3)
            
            # Make prediction
            y_pred = self.model.predict(test_image, verbose=0)
            y_pred_class = round(y_pred[0][0])
            
            prediction = self.class_names[y_pred_class]
            
            # Calculate confidence
            if y_pred <= 0.5:
                confidence = (1 - y_pred[0][0]) * 100
            else:
                confidence = y_pred[0][0] * 100
                
            return {
                "image_path": str(image_path),
                "filename": os.path.basename(image_path),
                "prediction": prediction,
                "confidence": float(round(confidence, 2)),
                "raw_score": float(y_pred[0][0])
            }
            
        except Exception as e:
            print(f"Prediction error for {image_path}: {str(e)}")
            return None

    def get_image_files(self, dataset_path):
        """Get all image files from dataset path"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        dataset_path = Path(dataset_path)
        
        if dataset_path.is_file():
            if dataset_path.suffix.lower() in image_extensions:
                image_files.append(dataset_path)
        elif dataset_path.is_dir():
            for ext in image_extensions:
                # Use case-insensitive search to avoid duplicates
                image_files.extend(dataset_path.rglob(f'*{ext}'))
                image_files.extend(dataset_path.rglob(f'*{ext.upper()}'))
        
        # Remove duplicates by converting to set and back to list
        unique_files = list(set(image_files))
        return sorted(unique_files)

    def process_batch(self, dataset_path, show_progress=True):
        """Process all images in the dataset"""
        start_time = time.time()
        
        # Get all image files
        image_files = self.get_image_files(dataset_path)
        
        if not image_files:
            print(f"No image files found in: {dataset_path}")
            return False
            
        print(f"Found {len(image_files)} image files")
        print("Starting batch processing...\n")
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            if show_progress:
                print(f"[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
            
            result = self.predict_single_image(image_path)
            
            if result:
                self.results.append(result)
                # Update stats
                if result["prediction"] == "Forged":
                    self.stats["forged_count"] += 1
                else:
                    self.stats["authentic_count"] += 1
            else:
                self.stats["errors"] += 1
                
            self.stats["total_images"] = len(image_files)
        
        self.stats["processing_time"] = float(round(time.time() - start_time, 2))
        return True

    def print_detailed_results(self):
        """Print detailed results for each image"""
        print("\n" + "="*80)
        print("📊 DETAILED RESULTS")
        print("="*80)
        
        for i, result in enumerate(self.results, 1):
            status_emoji = "🔴" if result["prediction"] == "Forged" else "🟢"
            print(f"{i:3d}. {status_emoji} {result['filename']:<30} | "
                  f"{result['prediction']:<10} | {result['confidence']:6.2f}%")

    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        
        total = self.stats["total_images"]
        forged = self.stats["forged_count"]
        authentic = self.stats["authentic_count"]
        errors = self.stats["errors"]
        
        print(f"📁 Total Images Processed: {total}")
        print(f"🔴 Forged Documents:       {forged} ({forged/total*100:.1f}%)")
        print(f"🟢 Authentic Documents:    {authentic} ({authentic/total*100:.1f}%)")
        if errors > 0:
            print(f"Processing Errors:      {errors}")
        print(f"⏱️  Processing Time:        {self.stats['processing_time']} seconds")

    def save_results_to_json(self, output_file):
        """Save results to JSON file"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "statistics": self.stats,
                "detailed_results": self.results
            }
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\nResults saved to: {output_file}")
            return True
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Batch Document Forgery Detection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_test.py /path/to/images
  python batch_test.py /path/to/images --model ./models/my_model.h5
  python batch_test.py /path/to/images --output results.json
  python batch_test.py single_image.jpg --quiet
        """
    )
    
    parser.add_argument('dataset_path', 
                       help='Path to dataset folder or single image file')
    parser.add_argument('--model', 
                       default='./models/trained_model.h5',
                       help='Path to trained model file (default: ./models/trained_model.h5)')
    parser.add_argument('--output', 
                       help='Save results to JSON file')
    parser.add_argument('--quiet', 
                       action='store_true',
                       help='Suppress detailed progress output')
    
    args = parser.parse_args()
    
    # Header
    print("="*80)
    print("🔍 BATCH DOCUMENT FORGERY DETECTION TOOL")
    print("="*80)
    
    # Check if dataset path exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path not found: {args.dataset_path}")
        sys.exit(1)
    
    # Initialize detector
    detector = BatchForgeryDetector(args.model)
    
    # Load model
    if not detector.load_model():
        sys.exit(1)
    
    # Process batch
    success = detector.process_batch(args.dataset_path, show_progress=not args.quiet)
    
    if not success:
        sys.exit(1)
    
    # Print results
    if not args.quiet:
        detector.print_detailed_results()
    
    detector.print_summary()
    
    # Save to file if requested
    if args.output:
        detector.save_results_to_json(args.output)
    
    print("\nBatch processing completed!")

if __name__ == "__main__":
    main()