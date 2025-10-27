"""
Image Loading and Preprocessing

Handles robust image loading with automatic format detection, normalization,
and preprocessing. Supports multiple image formats and provides consistent
float64 [0,1] normalized output for TDA pipeline.

Features:
- Multi-format support (PNG, TIFF, JPEG)
- Grayscale conversion
- Intensity normalization
- Error handling and validation

Author: Cheena Yadav
Date: October 2025
Version: 1.0.0
"""

import os
import h5py
import numpy as np
import cv2
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Iterator, Generator
from dataclasses import dataclass
import json
from PIL import Image
import tifffile
from tqdm import tqdm
import psutil
from datetime import datetime

from src.utils.logger import TDALogger, log_method_call


@dataclass
class DatasetMetadata:
    """
    Comprehensive metadata for dataset tracking and validation.
    
    This class captures all relevant information about a dataset
    for reproducibility and quality assurance.
    """
    name: str
    path: str
    total_files: int
    file_formats: List[str]
    total_size_bytes: int
    image_shapes: List[Tuple[int, ...]]
    data_types: List[str]
    color_modes: List[str]
    creation_time: str
    validation_status: str
    quality_metrics: Dict[str, Any]

@dataclass
class ImageMetadata:
    """Metadata for individual images."""
    filename: str
    filepath: str
    shape: Tuple[int, ...]
    dtype: str
    color_mode: str
    file_size_bytes: int
    format: str
    quality_score: float

class EnhancedDataLoader:
    """
    Production-grade data loader for TDA pipeline.
    
    Features:
    - Multi-format support with automatic detection
    - Memory-efficient streaming for large datasets
    - Comprehensive validation and quality assessment
    - Detailed logging and progress tracking
    - Flexible configuration management
    - Error recovery and graceful degradation
    """
    
    def __init__(self, 
                 logger: Optional[TDALogger] = None,
                 memory_limit_gb: float = 8.0,
                 enable_validation: bool = True,
                 enable_quality_check: bool = True):
        """
        Initialize the enhanced data loader.
        
        Args:
            logger: TDA logger instance for tracking
            memory_limit_gb: Maximum memory usage in GB
            enable_validation: Enable data validation
            enable_quality_check: Enable quality assessment
        """
        self.logger = logger or TDALogger(name="DataLoader")
        self.memory_limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)
        self.enable_validation = enable_validation
        self.enable_quality_check = enable_quality_check
        
        # Supported formats
        self.image_formats = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
        self.array_formats = {'.npy', '.npz'}
        self.scientific_formats = {'.h5', '.hdf5', '.mat'}
        
        self.logger.info("🔧 Enhanced Data Loader initialized")
        self.logger.info(f"💾 Memory limit: {memory_limit_gb:.1f} GB")


    @log_method_call
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and validate dataset configuration.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.logger.info(f"📋 Loading configuration: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        # Validate required fields
        required_fields = ['name', 'path']
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")
        
        # Set defaults
        config.setdefault('extensions', ['.png', '.jpg', '.jpeg'])
        config.setdefault('color_mode', 'grayscale')
        config.setdefault('recursive', False)
        config.setdefault('max_files', None)
        config.setdefault('validation_enabled', self.enable_validation)
        
        # Normalize extensions
        config['extensions'] = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                               for ext in config['extensions']]
        
        # Resolve and validate path
        dataset_path = Path(config['path'])
        if not dataset_path.is_absolute():
            # Try relative to config file directory
            dataset_path = config_path.parent / dataset_path
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        config['path'] = str(dataset_path.resolve())
        
        self.logger.info(f"✅ Configuration loaded successfully")
        self.logger.debug(f"📊 Config details: {json.dumps(config, indent=2)}")
        
        return config
    
    @log_method_call
    def discover_files(self, config: Dict[str, Any]) -> List[str]:
        """
        Discover all valid files in the dataset directory.
        
        Args:
            config: Dataset configuration
            
        Returns:
            List of absolute file paths
        """
        base_path = Path(config['path'])
        extensions = tuple(config['extensions'])
        recursive = config.get('recursive', False)
        max_files = config.get('max_files')
        
        self.logger.info(f"🔍 Discovering files in: {base_path}")
        self.logger.info(f"📁 Recursive search: {recursive}")
        self.logger.info(f"🎯 Target extensions: {extensions}")
        
        if recursive:
            pattern = "**/*"
            files = [p for p in base_path.rglob("*") 
                    if p.is_file() and p.suffix.lower() in extensions]
        else:
            files = [p for p in base_path.iterdir() 
                    if p.is_file() and p.suffix.lower() in extensions]
        
        # Sort for reproducibility
        files = sorted([str(p.resolve()) for p in files])
        
        # Apply file limit if specified
        if max_files and len(files) > max_files:
            self.logger.warning(f"⚠️  Limiting to {max_files} files (found {len(files)})")
            files = files[:max_files]
        
        if not files:
            raise FileNotFoundError(
                f"No files found with extensions {extensions} in {base_path}"
            )
        
        self.logger.info(f"✅ Discovered {len(files)} files")
        self.logger.debug(f"Discovered files: {files}")
        return files
    
    @log_method_call
    def load_image(self, 
                   filepath: str, 
                   color_mode: str = 'grayscale',
                   target_dtype: str = 'float32') -> Tuple[np.ndarray, ImageMetadata]:
        """
        Load a single image with comprehensive metadata extraction.
        
        Args:
            filepath: Path to image file
            color_mode: 'grayscale', 'rgb', or 'auto'
            target_dtype: Target numpy dtype
            
        Returns:
            Tuple of (image_array, metadata)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Image file not found: {filepath}")
        
        # Get file info
        file_size = filepath.stat().st_size
        file_format = filepath.suffix.lower()
        
        try:
            # Load based on format
            if file_format in {'.tiff', '.tif'}:
                image = tifffile.imread(str(filepath))
            elif file_format in {'.png', '.jpg', '.jpeg', '.bmp'}:
                if color_mode == 'grayscale':
                    image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
                elif color_mode == 'rgb':
                    image = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:  # auto
                    image = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
            else:
                # Fallback to PIL
                pil_image = Image.open(filepath)
                if color_mode == 'grayscale':
                    pil_image = pil_image.convert('L')
                elif color_mode == 'rgb':
                    pil_image = pil_image.convert('RGB')
                image = np.array(pil_image)
            
            if image is None:
                raise ValueError(f"Failed to load image: {filepath}")
            
            # Convert dtype if needed
            if target_dtype != str(image.dtype):
                if target_dtype == 'float32':
                    image = image.astype(np.float32)
                    if image.max() > 1.0:
                        image = image / 255.0
                elif target_dtype == 'uint8':
                    if image.dtype == np.float32 or image.dtype == np.float64:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
            
            # Calculate quality score
            quality_score = self._calculate_image_quality(image) if self.enable_quality_check else 1.0
            
            # Create metadata
            metadata = ImageMetadata(
                filename=filepath.name,
                filepath=str(filepath),
                shape=image.shape,
                dtype=str(image.dtype),
                color_mode=color_mode,
                file_size_bytes=file_size,
                format=file_format,
                quality_score=quality_score
            )
            
            self.logger.debug(f"📷 Loaded image: {filepath.name} {image.shape} {image.dtype}")
            
            return image, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load image {filepath}: {str(e)}")
            raise
    
    @log_method_call
    def load_array(self, filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load numpy array with metadata.
        
        Args:
            filepath: Path to .npy or .npz file
            
        Returns:
            Tuple of (array, metadata)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Array file not found: {filepath}")
        
        try:
            if filepath.suffix == '.npy':
                array = np.load(str(filepath))
                metadata = {
                    'filename': filepath.name,
                    'shape': array.shape,
                    'dtype': str(array.dtype),
                    'format': 'npy'
                }
            elif filepath.suffix == '.npz':
                npz_file = np.load(str(filepath))
                # For .npz files, return the first array and list all keys
                keys = list(npz_file.keys())
                array = npz_file[keys[0]]
                metadata = {
                    'filename': filepath.name,
                    'shape': array.shape,
                    'dtype': str(array.dtype),
                    'format': 'npz',
                    'available_keys': keys
                }
            else:
                raise ValueError(f"Unsupported array format: {filepath.suffix}")
            
            self.logger.debug(f"📊 Loaded array: {filepath.name} {array.shape} {array.dtype}")
            
            return array, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load array {filepath}: {str(e)}")
            raise
    
    @log_method_call
    def load_h5_dataset(self, 
                       filepath: str, 
                       dataset_key: str = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load data from HDF5 file.
        
        Args:
            filepath: Path to .h5 or .hdf5 file
            dataset_key: Specific dataset key (if None, loads first dataset)
            
        Returns:
            Tuple of (array, metadata)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"H5 file not found: {filepath}")
        
        try:
            with h5py.File(str(filepath), 'r') as f:
                # Get available datasets
                available_keys = list(f.keys())
                
                if dataset_key is None:
                    if not available_keys:
                        raise ValueError(f"No datasets found in {filepath}")
                    dataset_key = available_keys[0]
                    self.logger.info(f"🔑 Using dataset key: {dataset_key}")
                
                if dataset_key not in f:
                    raise KeyError(f"Dataset '{dataset_key}' not found in {filepath}")
                
                # Load dataset
                dataset = f[dataset_key]
                array = dataset[:]
                
                # Extract metadata
                metadata = {
                    'filename': filepath.name,
                    'dataset_key': dataset_key,
                    'shape': array.shape,
                    'dtype': str(array.dtype),
                    'format': 'h5',
                    'available_keys': available_keys,
                    'attributes': dict(dataset.attrs) if hasattr(dataset, 'attrs') else {}
                }
            
            self.logger.debug(f"🗃️  Loaded H5 dataset: {filepath.name}[{dataset_key}] {array.shape}")
            
            return array, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load H5 file {filepath}: {str(e)}")
            raise
    
    def stream_dataset(self, 
                      config: Dict[str, Any],
                      batch_size: int = 32) -> Generator[Tuple[List[np.ndarray], List[Dict]], None, None]:
        """
        Stream dataset in batches for memory-efficient processing.
        
        Args:
            config: Dataset configuration
            batch_size: Number of images per batch
            
        Yields:
            Tuples of (image_batch, metadata_batch)
        """
        files = self.discover_files(config)
        color_mode = config.get('color_mode', 'grayscale')
        
        self.logger.info(f"🌊 Starting dataset streaming: {len(files)} files, batch_size={batch_size}")
        self.logger.debug(f"Streaming files: {files}")
        
        batch_images = []
        batch_metadata = []
        
        with tqdm(files, desc="Loading dataset") as pbar:
            for filepath in pbar:
                try:
                    # Check memory usage
                    memory_usage = psutil.virtual_memory().used
                    if memory_usage > self.memory_limit_bytes:
                        self.logger.warning(f"⚠️  Memory limit exceeded: {memory_usage / 1024**3:.1f} GB")
                        # Yield current batch and clear memory
                        if batch_images:
                            yield batch_images, batch_metadata
                            batch_images = []
                            batch_metadata = []
                    
                    # Load image based on format
                    file_ext = Path(filepath).suffix.lower()
                    
                    if file_ext in self.image_formats:
                        image, metadata = self.load_image(filepath, color_mode)
                    elif file_ext in self.array_formats:
                        image, metadata = self.load_array(filepath)
                    elif file_ext in self.scientific_formats:
                        image, metadata = self.load_h5_dataset(filepath)
                    else:
                        self.logger.warning(f"⚠️  Unsupported format: {filepath}")
                        continue
                    
                    batch_images.append(image)
                    batch_metadata.append(metadata)
                    
                    # Yield batch when full
                    if len(batch_images) >= batch_size:
                        yield batch_images, batch_metadata
                        batch_images = []
                        batch_metadata = []
                    
                    pbar.set_postfix({
                        'memory': f"{psutil.virtual_memory().used / 1024**3:.1f}GB",
                        'batch': len(batch_images)
                    })
                    
                except Exception as e:
                    self.logger.error(f"❌ Error loading {filepath}: {str(e)}")
                    continue
        
        # Yield remaining images
        if batch_images:
            yield batch_images, batch_metadata
        
        self.logger.info("✅ Dataset streaming completed")
    
    def _calculate_image_quality(self, image: np.ndarray) -> float:
        """
        Calculate image quality score based on various metrics.
        
        Args:
            image: Input image array
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Ensure uint8 for quality metrics
            if gray.dtype != np.uint8:
                gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
            
            # Calculate various quality metrics
            
            # 1. Laplacian variance (sharpness)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
            
            # 2. Contrast (standard deviation)
            contrast_score = min(gray.std() / 64.0, 1.0)  # Normalize
            
            # 3. Brightness distribution
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist / hist.sum()
            brightness_score = 1.0 - abs(0.5 - (hist_norm * np.arange(256)).sum() / 256)
            
            # 4. Dynamic range
            dynamic_range = (gray.max() - gray.min()) / 255.0
            
            # Combine scores
            quality_score = (sharpness_score * 0.3 + 
                           contrast_score * 0.3 + 
                           brightness_score * 0.2 + 
                           dynamic_range * 0.2)
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception:
            return 0.5  # Default quality score if calculation fails
    
    @log_method_call
    def validate_dataset(self, config: Dict[str, Any]) -> DatasetMetadata:
        """
        Comprehensive dataset validation and metadata extraction.
        
        Args:
            config: Dataset configuration
            
        Returns:
            Complete dataset metadata
        """
        self.logger.info("🔍 Starting comprehensive dataset validation")
        
        files = self.discover_files(config)
        
        # Initialize tracking variables
        total_size = 0
        file_formats = set()
        image_shapes = []
        data_types = set()
        color_modes = set()
        quality_scores = []
        validation_errors = []
        
        # Process each file
        for filepath in tqdm(files, desc="Validating dataset"):
            try:
                file_path = Path(filepath)
                file_size = file_path.stat().st_size
                total_size += file_size
                
                file_format = file_path.suffix.lower()
                file_formats.add(file_format)
                
                # Load and validate based on format
                if file_format in self.image_formats:
                    image, metadata = self.load_image(filepath, config.get('color_mode', 'grayscale'))
                    image_shapes.append(image.shape)
                    data_types.add(str(image.dtype))
                    color_modes.add(metadata.color_mode)
                    quality_scores.append(metadata.quality_score)
                    
                elif file_format in self.array_formats:
                    array, metadata = self.load_array(filepath)
                    image_shapes.append(array.shape)
                    data_types.add(str(array.dtype))
                    
                elif file_format in self.scientific_formats:
                    array, metadata = self.load_h5_dataset(filepath)
                    image_shapes.append(array.shape)
                    data_types.add(str(array.dtype))
                
            except Exception as e:
                validation_errors.append(f"{filepath}: {str(e)}")
                self.logger.warning(f"⚠️  Validation error for {filepath}: {str(e)}")
        
        # Calculate quality metrics
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        quality_metrics = {
            "average_quality": avg_quality,
            "quality_std": np.std(quality_scores) if quality_scores else 0.0,
            "min_quality": min(quality_scores) if quality_scores else 0.0,
            "max_quality": max(quality_scores) if quality_scores else 0.0,
            "validation_errors": len(validation_errors),
            "error_rate": len(validation_errors) / len(files) if files else 0.0
        }
        
        # Determine validation status
        if len(validation_errors) == 0:
            validation_status = "PASSED"
        elif len(validation_errors) / len(files) < 0.1:  # Less than 10% errors
            validation_status = "PASSED_WITH_WARNINGS"
        else:
            validation_status = "FAILED"
        
        # Create comprehensive metadata
        metadata = DatasetMetadata(
            name=config.get('name', 'Unknown'),
            path=config['path'],
            total_files=len(files),
            file_formats=list(file_formats),
            total_size_bytes=total_size,
            image_shapes=image_shapes,
            data_types=list(data_types),
            color_modes=list(color_modes),
            creation_time=str(datetime.now()),
            validation_status=validation_status,
            quality_metrics=quality_metrics
        )
        
        self.logger.info(f"✅ Dataset validation completed: {validation_status}")
        self.logger.info(f"📊 Files: {len(files)}, Size: {total_size / 1024**2:.1f} MB")
        self.logger.info(f"🎯 Average quality: {avg_quality:.3f}")
        
        if validation_errors:
            self.logger.warning(f"⚠️  {len(validation_errors)} validation errors found")
        
        return metadata

# Convenience functions for backward compatibility
def load_config(config_path: str) -> Dict[str, Any]:
    """Load dataset configuration using enhanced loader."""
    loader = EnhancedDataLoader()
    return loader.load_config(config_path)

def load_dataset(config_path: str) -> Tuple[List[np.ndarray], List[str]]:
    """Load complete dataset into memory."""
    loader = EnhancedDataLoader()
    config = loader.load_config(config_path)
    
    images = []
    paths = []
    
    for batch_images, batch_metadata in loader.stream_dataset(config, batch_size=1000):
        images.extend(batch_images)
        paths.extend([meta.filepath for meta in batch_metadata])
    
    return images, paths
