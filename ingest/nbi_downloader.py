import os
import re
import zipfile
import requests

BASE_NBI_URL = "https://www.fhwa.dot.gov/bridge/nbi/"


def get_download_link(year: int, delimited: bool = True) -> str:
    """
    Fetch the FHWA disclaimer page and extract the direct ZIP download link.
    Tries delimited first, falls back to non-delimited (raw ASCII) for older years.
    """
    suffix = "del" if delimited else ""
    url = f"{BASE_NBI_URL}disclaim.cfm?nbiYear={year}{suffix}&nbiZip=zip"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            # Look for links ending in .zip
            matches = re.findall(r'href="([^"]+\.zip)"', response.text)
            if matches:
                # Resolve relative URL
                zip_file = matches[0]
                return BASE_NBI_URL + zip_file
    except Exception as e:
        print(f"[WARN] Error fetching disclaimer for year {year}: {e}")

    # Fallback to non-delimited if delimited was requested but failed
    if delimited:
        print(f"[INFO] Delimited download failed for {year}, trying raw ASCII...")
        return get_download_link(year, delimited=False)
        
    raise ValueError(f"Could not resolve NBI download link for year {year}")


def download_nbi_year(year: int, output_dir: str = "data/raw") -> str:
    """
    Downloads NBI ZIP file for the specified year, saves it, and unzips it.
    Returns the path to the directory containing unzipped files.
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, f"nbi_{year}.zip")
    extract_path = os.path.join(output_dir, f"nbi_{year}")

    # If extracted folder already exists, skip download
    if os.path.exists(extract_path) and os.listdir(extract_path):
        print(f"[OK] NBI files for {year} already extracted at {extract_path}")
        return extract_path

    # Get the link
    download_url = get_download_link(year)
    print(f"[INFO] Downloading {year} NBI from: {download_url}")
    
    response = requests.get(download_url, stream=True, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download NBI for {year}. HTTP {response.status_code}")
        
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                
    print(f"[OK] Downloaded {year} ZIP to {zip_path}")
    
    # Extract zip
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
        
    print(f"[OK] Extracted NBI files for {year} to {extract_path}")
    
    # Clean up the zip file to save space
    try:
        os.remove(zip_path)
    except OSError:
        pass
        
    return extract_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target_year = int(sys.argv[1])
        download_nbi_year(target_year)
    else:
        print("Usage: python ingest/nbi_downloader.py <year>")
