# Riftbound Card Tracker

Switch to 'src/db/'. Execute scraper.py. A SQLite Database will be initialized, card 
data will be scraped and written into the DB. Second the picture_loader.py downloads 
the pictures from the URLs which are in the DB under the column 'image' on your local
machine. Then in 'src/trans/' execute transform.py. This script converts and resizes
the pictures. This is mandatory for future use in CNNs.

The commands for the terminal:

1. python -m src.db.scraper (optional)
2. python -m src.db.picture_loader
3. python -m src.trans.transform