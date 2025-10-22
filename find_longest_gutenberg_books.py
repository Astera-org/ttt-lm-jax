#!/usr/bin/env python3
"""
Script to find the longest books from Project Gutenberg.
Downloads book metadata, analyzes lengths, and identifies the longest books.
"""

import requests
import time
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import re
import pickle

# Directory to store metadata and results
DATA_DIR = Path("./data/gutenberg")
CACHE_DIR = DATA_DIR / "cache"
METADATA_FILE = DATA_DIR / "book_metadata.json"
DEFAULT_RESULTS_FILENAME = "longest_{top_n}_books.json"
CACHE_FILE = CACHE_DIR / "book_lengths_cache.pkl"

def load_cache() -> Dict[int, Tuple[str, int]]:
    """Load cached book lengths from disk."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
            print(f"Loaded cache with {len(cache)} books")
            return cache
        except Exception as e:
            print(f"Error loading cache: {e}")
            return {}
    return {}

def save_cache(cache: Dict[int, Tuple[str, int]]):
    """Save book lengths cache to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
        print(f"Saved cache with {len(cache)} books")
    except Exception as e:
        print(f"Error saving cache: {e}")

def get_book_length(book_id: int, cache: Optional[Dict[int, Tuple[str, int]]] = None) -> Optional[Tuple[int, str, int]]:
    """
    Get the length of a book by downloading its text.
    Returns (book_id, title, character_count) or None if not available.
    Uses cache if provided.
    """
    # Check cache first
    if cache is not None and book_id in cache:
        title, char_count = cache[book_id]
        return (book_id, title, char_count)
    
    try:
        # Try to get the plain text version
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            # Try alternative URL format
            url = f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt"
            response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            text = response.text
            # Try to extract title from the header
            title_match = re.search(r'Title:\s*(.+)', text)
            title = title_match.group(1).strip() if title_match else f"Book {book_id}"
            
            # Remove Project Gutenberg header and footer
            # Header typically ends with "*** START OF"
            # Footer typically starts with "*** END OF"
            start_match = re.search(r'\*\*\* START OF (THIS|THE) PROJECT GUTENBERG', text, re.IGNORECASE)
            end_match = re.search(r'\*\*\* END OF (THIS|THE) PROJECT GUTENBERG', text, re.IGNORECASE)
            
            if start_match and end_match:
                actual_text = text[start_match.end():end_match.start()]
            else:
                actual_text = text
            
            char_count = len(actual_text.strip())
            
            # Update cache
            if cache is not None:
                cache[book_id] = (title, char_count)
            
            return (book_id, title, char_count)
        else:
            return None
            
    except Exception as e:
        return None

def scan_all_books(max_book_id: int = 75000, cache: Optional[Dict[int, Tuple[str, int]]] = None) -> List[Tuple[int, str, int]]:
    """
    Scan all books from Project Gutenberg (up to max_book_id).
    Uses cache to avoid re-downloading books.
    """
    if cache is None:
        cache = load_cache()
    
    print(f"Scanning books 1 to {max_book_id}...")
    print(f"Already have {len(cache)} books in cache")
    
    # Determine which books need to be checked
    all_book_ids = list(range(1, max_book_id + 1))
    books_to_check = [bid for bid in all_book_ids if bid not in cache]
    
    print(f"Need to check {len(books_to_check)} new books")
    
    results = []
    failed = 0
    save_interval = 100  # Save cache every 100 books
    
    # Add cached books to results
    for book_id, (title, char_count) in cache.items():
        if book_id <= max_book_id:
            results.append((book_id, title, char_count))
    
    print(f"Starting with {len(results)} books from cache")
    
    # Check new books
    for idx, book_id in enumerate(tqdm(books_to_check, desc="Checking new books"), 1):
        result = get_book_length(book_id, cache)
        if result:
            results.append(result)
        else:
            failed += 1
        
        # Save cache periodically
        if idx % save_interval == 0:
            save_cache(cache)
        
        # Be nice to the server
        time.sleep(0.15)
    
    # Final cache save
    save_cache(cache)
    
    print(f"\nTotal: {len(results)} books found, {failed} failed")
    return results

def find_longest_books(sample_results: List[Tuple[int, str, int]], top_n: int = 200) -> List[Dict]:
    """
    Find the longest books from the sample.
    """
    # Sort by character count (descending)
    sorted_books = sorted(sample_results, key=lambda x: x[2], reverse=True)
    
    # Take the top N
    longest_books = sorted_books[:top_n]
    
    # Convert to dict format
    result = []
    for rank, (book_id, title, char_count) in enumerate(longest_books, 1):
        result.append({
            "rank": rank,
            "book_id": book_id,
            "title": title,
            "character_count": char_count,
            "url": f"https://www.gutenberg.org/ebooks/{book_id}"
        })
    
    return result

def save_results(longest_books: List[Dict], output_file: Path):
    """Save results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(longest_books, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")

def print_results(longest_books: List[Dict]):
    """Print the results in a nice format."""
    print("\n" + "="*80)
    print(f"TOP {len(longest_books)} LONGEST BOOKS FROM PROJECT GUTENBERG")
    print("="*80 + "\n")
    
    for book in longest_books:
        print(f"{book['rank']:2d}. [{book['book_id']:5d}] {book['title'][:60]:60s} - {book['character_count']:,} chars")
    
    print("\n" + "="*80)
    print(f"Total: {len(longest_books)} books")
    print("="*80)

def download_longest_books(longest_books: List[Dict], output_dir: Path):
    """
    Download the actual text files of the longest books.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading {len(longest_books)} books to {output_dir}...")
    
    for book in tqdm(longest_books, desc="Downloading books"):
        book_id = book['book_id']
        filename = f"{book_id}_{book['title'][:50].replace('/', '_').replace(' ', '_')}.txt"
        filepath = output_dir / filename
        
        try:
            # Try to get the plain text version
            url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                url = f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt"
                response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"  ✓ Downloaded: {book['title'][:50]}")
            else:
                print(f"  ✗ Failed to download: {book['title'][:50]}")
        
        except Exception as e:
            print(f"  ✗ Error downloading {book['title'][:50]}: {e}")
        
        # Be nice to the server
        time.sleep(0.5)

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Find the longest books from Project Gutenberg")
    parser.add_argument("--max-book-id", type=int, default=75000, 
                       help="Maximum book ID to check (default: 75000)")
    parser.add_argument("--top-n", type=int, default=200,
                       help="Number of longest books to find (default: 200)")
    parser.add_argument("--download", action="store_true",
                       help="Download the actual book files")
    parser.add_argument("--output-dir", type=str, default="./data/gutenberg",
                       help="Output directory for downloaded books")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear the cache and start fresh")
    parser.add_argument("--cache-only", action="store_true",
                       help="Use existing cache without downloading new books")
    
    args = parser.parse_args()

    # Clear cache if requested
    if args.clear_cache:
        if CACHE_FILE.exists():
            os.remove(CACHE_FILE)
            print("Cache cleared")
    
    # Load cache
    cache = load_cache()

    if args.cache_only:
        book_data = [
            (book_id, title, char_count)
            for book_id, (title, char_count) in cache.items()
            if book_id <= args.max_book_id
        ]
        print(f"Using {len(book_data)} books from cache only")
    else:
        # Scan all books
        book_data = scan_all_books(args.max_book_id, cache)
    
    # Find the longest books
    longest_books = find_longest_books(book_data, args.top_n)
    
    # Print results
    print_results(longest_books)
    
    # Save results
    output_dir = Path(args.output_dir)
    results_filename = DEFAULT_RESULTS_FILENAME.format(top_n=args.top_n)
    save_results(longest_books, output_dir / results_filename)
    
    # Download books if requested
    if args.download:
        download_longest_books(longest_books, output_dir)
    else:
        print("\nTo download these books, run with --download flag")
        print(f"Example: python {__file__} --download")

if __name__ == "__main__":
    main()
