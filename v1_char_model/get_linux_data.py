import requests
import time

# List of C files to download from the Linux kernel
FILES_TO_DOWNLOAD = [
    'kernel/sched/core.c',
    'mm/memory.c',
    'fs/open.c',
    'kernel/printk/printk.c'
]

BASE_URL = 'https://raw.githubusercontent.com/torvalds/linux/master/'
OUTPUT_FILE = 'input.txt'

def download_file(filepath):
    """Download a single file from the Linux kernel repository."""
    url = BASE_URL + filepath
    print(f"Downloading {filepath}...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filepath}: {e}")
        return None

def main():
    total_size = 0
    all_content = []
    
    print("Starting download of Linux kernel C files...")
    print(f"Target: ~1MB of C code\n")
    
    for filepath in FILES_TO_DOWNLOAD:
        content = download_file(filepath)
        
        if content:
            # Wrap content with file boundary markers
            wrapped_content = f"/* --- START OF {filepath} --- */\n"
            wrapped_content += content
            wrapped_content += f"\n/* --- END OF {filepath} --- */\n\n"
            
            all_content.append(wrapped_content)
            file_size = len(content.encode('utf-8'))
            total_size += file_size
            
            print(f"  ✓ Downloaded {filepath}: {file_size:,} bytes")
        
        # Be nice to GitHub's servers
        time.sleep(0.5)
    
    # Write all content to input.txt
    if all_content:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(''.join(all_content))
        
        print(f"\n{'='*60}")
        print(f"SUCCESS!")
        print(f"Total size: {total_size:,} bytes ({total_size / (1024*1024):.2f} MB)")
        print(f"Files downloaded: {len(all_content)}")
        print(f"Output saved to: {OUTPUT_FILE}")
        print(f"{'='*60}")
    else:
        print("ERROR: No files were downloaded successfully.")

if __name__ == '__main__':
    main()
