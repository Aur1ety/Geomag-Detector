import os
import re
import requests

def download_from_sh(sh_path):
    # Base PRADAN URL
    base_url = "https://pradan.issdc.gov.in"
    
    if not os.path.exists(sh_path):
        print(f"Error: {sh_path} not found.")
        return

    with open(sh_path, 'r') as f:
        content = f.read()

    # Find all download links in the script (usually starts with /al1/...)
    links = re.findall(r'(/al1/protected/downloadData/.*?\.nc\?mag)', content)
    
    if not links:
        print("No valid download links found in the .sh file.")
        return

    print(f"Found {len(links)} files to download. Starting...")

    for link in links:
        full_url = base_url + link
        # Extract filename (e.g., L2_AL1_MAG_20250815_V00.nc)
        filename = re.search(r'(L2_AL1_MAG_.*?\.nc)', link).group(1)
        
        print(f"Downloading {filename}...", end="\r")
        
        response = requests.get(full_url, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            print(f"\nFailed: {filename} (Status: {response.status_code})")
            print("Check if your PRADAN session has expired again.")
            break

    print("\nDownload process complete.")

# Run for your specific folders
# Change the path to just the filename since you're already in the folder
download_from_sh("mag_2026May15T032816491.sh")