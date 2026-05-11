"""
Universal data preprocessor for Fragment_LLM.

Supports any format, any size — processes by streaming so a 1 TB JSONL
file uses the same RAM as a 1 MB text file.

Supported input formats
-----------------------
  .txt     Plain UTF-8 text files
  .jsonl   JSON Lines — one JSON object per line, text extracted from a key
  .json    JSON array of objects, or single object with a text key
  .csv     CSV/TSV with a text column
  .parquet Apache Parquet (WikiText-style, HuggingFace datasets export)

Output
------
  data/processed/train.txt
  data/processed/val.txt
  data/processed/test.txt

Usage
-----
  python scripts/preprocessor.py
  python scripts/preprocessor.py --raw-dir data/raw --text-key text
  python scripts/preprocessor.py --raw-dir /big/corpus --min-chars 50 --dedup
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Character budget per output chunk written at once (10 MB)
_WRITE_CHUNK = 10_000_000


# ── Path / file helpers ────────────────────────────────────────────────────────

def _validate_dir(path_str: str, create: bool = False) -> Path:
    if ".." in path_str:
        raise ValueError(f"Path traversal detected: {path_str!r}")
    path = Path(path_str).resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    return path


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 ** 2


# ── Per-format streaming text generators ──────────────────────────────────────

def _stream_txt(path: Path, chunk_bytes: int = 4_096) -> Generator[str, None, None]:
    """Yield lines from a plain text file without loading it all."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line


def _stream_jsonl(path: Path, text_key: str) -> Generator[str, None, None]:
    """Yield text fields from a JSON Lines file, one record per line."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning(f"{path.name}:{lineno} — JSON decode error: {exc}")
                continue
            if isinstance(obj, dict):
                text = obj.get(text_key, "")
                if text:
                    yield str(text)
            elif isinstance(obj, str):
                yield obj


def _stream_json(path: Path, text_key: str) -> Generator[str, None, None]:
    """
    Yield text from a JSON file.
    Handles: array of objects, single object, or array of strings.
    Falls back to JSONL streaming if the file is too large to load at once (> 512 MB).
    """
    size_mb = _file_size_mb(path)
    if size_mb > 512:
        logger.info(f"{path.name} is {size_mb:.0f} MB — streaming as JSONL")
        yield from _stream_jsonl(path, text_key)
        return

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            logger.warning(f"JSON load failed ({exc}), retrying as JSONL")
            yield from _stream_jsonl(path, text_key)
            return

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and text_key in item:
                yield str(item[text_key])
            elif isinstance(item, str):
                yield item
    elif isinstance(data, dict):
        if text_key in data:
            yield str(data[text_key])
        else:
            for v in data.values():
                if isinstance(v, str):
                    yield v


def _stream_csv(path: Path, text_key: str) -> Generator[str, None, None]:
    """Yield text column from a CSV/TSV file using pandas chunking."""
    import pandas as pd
    sep = "\t" if path.suffix == ".tsv" else ","
    try:
        for chunk in pd.read_csv(path, sep=sep, usecols=[text_key],
                                  chunksize=10_000, engine="c",
                                  on_bad_lines="skip", low_memory=True):
            for val in chunk[text_key].dropna():
                yield str(val)
    except (ValueError, KeyError):
        logger.warning(f"Column {text_key!r} not found in {path.name}; "
                       f"available: trying all string columns")
        for chunk in pd.read_csv(path, sep=sep, chunksize=10_000,
                                  engine="c", on_bad_lines="skip"):
            str_cols = chunk.select_dtypes(include="object").columns
            if not len(str_cols):
                continue
            col = str_cols[0]
            logger.info(f"Using column {col!r} from {path.name}")
            for val in chunk[col].dropna():
                yield str(val)
            return


def _stream_parquet(path: Path, text_key: str) -> Generator[str, None, None]:
    """Yield text column from a Parquet file; uses row-group streaming."""
    import pandas as pd
    import pyarrow.parquet as pq  # noqa: F401 — already in requirements

    table = pq.read_table(path, columns=None)
    col = text_key if text_key in table.schema.names else None
    if col is None:
        # pick first string column
        import pyarrow as pa
        for field in table.schema:
            if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                col = field.name
                logger.info(f"Parquet: using column {col!r} from {path.name}")
                break
    if col is None:
        logger.error(f"No text column found in {path.name}")
        return

    # Stream row-group by row-group
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        batch = pf.read_row_group(rg, columns=[col]).to_pydict()[col]
        for val in batch:
            if val:
                yield str(val)


# ── Text sanitisation ─────────────────────────────────────────────────────────

def _sanitize(line: str, min_chars: int) -> Optional[str]:
    """Clean a single line of text.  Returns None to discard it."""
    line = line.replace("\x00", "")          # null bytes
    line = " ".join(line.split())            # normalise whitespace
    if not line or len(line) < min_chars:
        return None
    if line.startswith("=" * 5):             # WikiText section markers
        return None
    return line


# ── Main Preprocessor ─────────────────────────────────────────────────────────

class Preprocessor:
    """
    Universal streaming preprocessor.

    Discovers files for train/val/test splits in *raw_dir*, processes
    them through a streaming pipeline, and writes clean text to
    *processed_dir*.  The entire operation uses O(write_chunk) RAM
    regardless of input size.
    """

    SUPPORTED_EXT = {".txt", ".jsonl", ".json", ".csv", ".tsv", ".parquet"}

    def __init__(
        self,
        raw_dir:       str = "data/raw",
        processed_dir: str = "data/processed",
        text_key:      str = "text",         # JSON/CSV column name
        min_chars:     int = 10,             # discard lines shorter than this
        dedup:         bool = False,         # de-duplicate lines (uses RAM for bloom set)
        max_file_size_mb: int = 0,           # 0 = no limit
    ) -> None:
        self.raw_dir       = _validate_dir(raw_dir, create=False)
        self.processed_dir = _validate_dir(processed_dir, create=True)
        self.text_key      = text_key
        self.min_chars     = min_chars
        self.dedup         = dedup
        self.max_file_size_mb = max_file_size_mb
        self._seen: set = set()  # for dedup (stores SHA-256 prefixes)

        logger.info(f"Preprocessor  raw={self.raw_dir}  out={self.processed_dir}  "
                    f"text_key={text_key!r}  dedup={dedup}")

    # ── routing ───────────────────────────────────────────────────────────

    def _stream_file(self, path: Path) -> Generator[str, None, None]:
        """Route to the correct streamer based on file extension."""
        ext = path.suffix.lower()
        if ext == ".txt":
            yield from _stream_txt(path)
        elif ext == ".jsonl":
            yield from _stream_jsonl(path, self.text_key)
        elif ext == ".json":
            yield from _stream_json(path, self.text_key)
        elif ext in (".csv", ".tsv"):
            yield from _stream_csv(path, self.text_key)
        elif ext == ".parquet":
            yield from _stream_parquet(path, self.text_key)
        else:
            logger.warning(f"Unsupported extension {ext!r}: {path.name} — skipping")

    # ── discovery ─────────────────────────────────────────────────────────

    def _find_files(self, pattern: str) -> List[Path]:
        """Find files matching *pattern* in any supported format."""
        found = []
        for ext in self.SUPPORTED_EXT:
            found.extend(self.raw_dir.glob(f"**/{pattern}*{ext}"))
        found.sort()
        return found

    # ── dedup ─────────────────────────────────────────────────────────────

    def _is_duplicate(self, line: str) -> bool:
        if not self.dedup:
            return False
        key = hashlib.sha256(line[:200].encode()).digest()[:8]
        if key in self._seen:
            return True
        self._seen.add(key)
        return False

    # ── process one split ─────────────────────────────────────────────────

    def _process_split(self, search_pattern: str, output_name: str) -> bool:
        files = self._find_files(search_pattern)
        if not files:
            logger.warning(f"No files found for split {output_name!r} "
                           f"(pattern: {search_pattern!r})")
            return False

        out_path = self.processed_dir / f"{output_name}.txt"
        total_lines = 0
        total_bytes = 0

        logger.info(f"Processing {output_name!r}: {[f.name for f in files]}")

        with open(out_path, "w", encoding="utf-8") as fout:
            buf: List[str] = []
            buf_chars = 0

            for path in files:
                size_mb = _file_size_mb(path)
                if self.max_file_size_mb and size_mb > self.max_file_size_mb:
                    logger.warning(f"Skipping {path.name} ({size_mb:.0f} MB > "
                                   f"limit {self.max_file_size_mb} MB)")
                    continue

                logger.info(f"  ← {path.name}  ({size_mb:.1f} MB)")

                for raw_line in self._stream_file(path):
                    clean = _sanitize(raw_line, self.min_chars)
                    if clean is None:
                        continue
                    if self._is_duplicate(clean):
                        continue
                    buf.append(clean)
                    buf_chars += len(clean)
                    total_lines += 1

                    if buf_chars >= _WRITE_CHUNK:
                        chunk = "\n".join(buf) + "\n"
                        fout.write(chunk)
                        total_bytes += len(chunk.encode())
                        buf.clear()
                        buf_chars = 0

            # flush remainder
            if buf:
                chunk = "\n".join(buf) + "\n"
                fout.write(chunk)
                total_bytes += len(chunk.encode())

        out_mb = total_bytes / 1024**2
        logger.info(f"  → {out_path.name}: {total_lines:,} lines, {out_mb:.2f} MB")
        return True

    # ── public API ────────────────────────────────────────────────────────

    def preprocess(self) -> None:
        """
        Process train / val / test splits.
        Searches raw_dir for files whose name starts with 'train', 'val*'
        (or 'validation'), and 'test'.
        """
        logger.info("=" * 60)
        logger.info("Starting preprocessing …")

        splits = [
            ("train",      "train"),
            ("val",        "val"),
            ("validation", "val"),  # HuggingFace naming
            ("test",       "test"),
        ]

        seen_outputs: set = set()
        processed = 0
        for pattern, output in splits:
            if output in seen_outputs:
                continue
            if self._process_split(pattern, output):
                seen_outputs.add(output)
                processed += 1

        logger.info("=" * 60)
        logger.info(f"Preprocessing complete — {processed} split(s) written to "
                    f"{self.processed_dir}")
        logger.info("=" * 60)

    # Keep old name for backward compatibility
    def preprocess_wkitext(self) -> None:
        self.preprocess()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fragment_LLM universal preprocessor")
    parser.add_argument("--raw-dir",        default="data/raw",       help="Input directory")
    parser.add_argument("--processed-dir",  default="data/processed", help="Output directory")
    parser.add_argument("--text-key",       default="text",
                        help="JSON/CSV column that contains the text (default: 'text')")
    parser.add_argument("--min-chars",      type=int, default=10,
                        help="Discard lines shorter than N characters")
    parser.add_argument("--dedup",          action="store_true",
                        help="Remove duplicate lines (uses memory proportional to unique lines)")
    parser.add_argument("--max-file-size",  type=int, default=0,
                        help="Skip files larger than N MB (0 = no limit)")
    args = parser.parse_args()

    preprocessor = Preprocessor(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        text_key=args.text_key,
        min_chars=args.min_chars,
        dedup=args.dedup,
        max_file_size_mb=args.max_file_size,
    )
    preprocessor.preprocess()


if __name__ == "__main__":
    main()