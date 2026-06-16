import os
import requests
# Own Modules
from database import extract_id_image

# Extract IDs and Image URLs
rows = extract_id_image()

# Create Output Directory for Pictures
output_dir = "data/card_images"
os.makedirs(output_dir, exist_ok=True)

# Download Pictures Loop
i = 0
for row in rows:
    i += 1

    id, url = row[0], row[1]
    file_path = os.path.join(output_dir, f"{id}.png")
    if os.path.exists(file_path):
        continue
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        # Feedback for every 10th Download
        if i % 10 == 0:
            print(f"{i} Pictures downloaded!")
