"""
Data preprocessor for WikiText dataset
"""
import pandas as pd
from pathlib import Path
import logging
from typing import List, Tuple


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Preprocessor:
    """Data preprocessing with validation."""
    def __init__(
        self,
        raw_dir:str = "data/raw",
        processed_dir: str = "data/processed",
        max_file_size_mb: int = 500,
        allowed_extensions: Tuple[str, ...] = ('.parquet', '.txt')
    ):
        """Initialize preprocessor."""
        self.raw_dir = self._validate_directory(raw_dir, create=False)
        self.processed_dir = self._validate_directory(processed_dir, create=True)  
        self.max_file_size_mb = max_file_size_mb
        self.allowed_extensions = allowed_extensions
        
        logger.info(f"Preprocessor initialized")
        logger.info(f"Raw: {self.raw_dir}")
        logger.info(f"Processed: {self.processed_dir}")
        
    @staticmethod
    def _validate_directory(
        dir_path: str,
        create: bool = False
    ) -> Path:
        """
        Validate directory path with security checks.
        
        Args:
            dir_path: Directory path to validate
            create: Whether to create directory if it doesn't exist
        
        Returns:
            Validated Path object
        """
        try:
            path = Path(dir_path).resolve()
        except Exception as e:
            raise ValueError(f"Invalid directory path: {e}")
        
        if create:
            path.mkdir(exist_ok=True, parents=True)
            logger.info(f"Created directory: {path}")
        elif not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        
        return path
    
    def _validate_file(
        self,
        file_path: Path
    ) -> bool:
        """
        Validate file before processing.
        
        Args:
            file_path: Path to file
        
        Returns:
            True if file is valid
        """
        # Check file exists
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return False
        
        # Check file extension
        if file_path.suffix not in self.allowed_extensions:
            logger.warning(
                f"Invalid extension {file_path.suffix}: {file_path.name}"
            )
            return False

        return True
    
    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize text content.
        
        Args:
            text: Input text
        
        Returns:
            Sanitized text
        """
        # Remove null bytes (security)
        text = text.replace('\x00','')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _process_parquet_file(
        self,
        file_path: Path,
        text_column: str = 'text'
    ) -> List[str]:
        """
        Process a parquet file securely.
        
        Args:
            file_path: Path to parquet file
            text_column: Name of text column
        
        Returns:
            List of cleaned text lines
        """
        try:
            # Read parquet with size limit validation
            df = pd.read_parquet(file_path)
            
            # Validate column exists
            if text_column not in df.columns:
                available_cols = ', '.join(df.columns)
                raise KeyError(
                    f"Column '{text_column}' not found. "
                    f"Available columns: {available_cols}"
                )
                
            # Extract and validate text
            lines = df[text_column].astype(str).tolist()
            
            
            # Clean and filter lines
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and section markers
                if not line or line.startswith("=" * 10):
                    continue
                
                # Snitize
                line = self._sanitize_text(line)
                
                if line: # Ony add non-empty after sanitization
                    cleaned_lines.append(line)
            
            logger.info(
                f"Processed {file_path.name}: "
                f"{len(lines)} -> {len(cleaned_lines)}"
            )
            
            return cleaned_lines
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            raise
    
    def _process_text_file(self, file_path: Path) -> List[str]:
        """
        Process a text file securely.
        
        Args:
            file_path: Path to text file
        
        Returns:
            List of cleaned text lines
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            
            # Split into lines and clean
            lines = text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                
                if not line or line.startswith("=" * 10):
                    continue
            
                line = self._sanitize_text(line)
                
                if line:
                    cleaned_lines.append(line)
            
            logger.info(
                f"Processed {file_path.name}: "
                f"{len(lines)} -> {len(cleaned_lines)} lines"
            )
            
            return cleaned_lines
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            raise
            
    
    def preprocess_wkitext(self) -> None:
        """
        Preprocess WikiText dataset with security validation.
        Searches for train/validation/test splits in parquet or text format.
        """
        logger.info("Starting WikiText preprocessing...")
        
        # Define splits to process
        splits = [
            ('train', 'train'),
            ('validation', 'val'),
            ('test','test')
        ]
        
        processed_count = 0
        
        for search_pattern, output_name in splits:
            logger.info(F"\nProcessing {output_name} split...")
            
            # Search for parquet files first
            parquet_files = list(
                self.raw_dir.glob(f"**/{search_pattern}*.parquet")
            )
            
            # Fallback to text files
            text_files = list(
                self.raw_dir.glob(f"**/{search_pattern}*.txt")
            )
            
            files_to_process = parquet_files if parquet_files else text_files
            
            if not files_to_process:
                logger.warning(
                    f"No files found for {output_name} split"
                    f"(pattern: {search_pattern})"
                )
                continue
            
            # Process first valid file
            processed = False
            
            for file_path in files_to_process:
                if not self._validate_file(file_path):
                    continue
                
                try:
                    # Process based on file type
                    if file_path.suffix == '.parquet':
                        cleaned_lines = self._process_parquet_file(file_path)
                    else:
                        cleaned_lines = self._process_text_file(file_path)

                    # Save output
                    output_file = self.processed_dir / f"{output_name}.txt"
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(cleaned_lines))
                    
                    logger.info(
                        f"Saved {output_name}.txt: "
                        f"{len(cleaned_lines)} lines, "
                        f"{output_file.stat().st_size / (1024*1024):.2f} MB"
                    )
                    
                    processed = True
                    processed_count += 1
                    break
                
                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")
                    continue
            
            if not processed:
                logger.warning(f"No valid files processed for {output_name} split")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Preprocessing complete!")
        logger.info(f"Processed {processed_count} splits")
        logger.info(f"Output directory: {self.processed_dir}")
        logger.info(f"{'='*60}")
        
def main():
    """Main entry point for preprocessing."""
    try:
        preprocessor = Preprocessor(
            raw_dir='data/raw',
            processed_dir='data/processed',
            max_file_size_mb=500
        )
        
        preprocessor.preprocess_wkitext()
    
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise
    
if __name__ == "__main__":
    main()