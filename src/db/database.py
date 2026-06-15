import sqlite3

def init_db():
    conn = sqlite3.connect('riftbound.db')
    cursor = conn.cursor()

    cursor.execute('''
                CREATE TABLE IF NOT EXISTS cards (
                   id TEXT PRIMARY KEY,
                   name TEXT,
                   collectorNumber TEXT,
                   publicCode TEXT,
                   setName TEXT,
                   cardType TEXT,
                   superType TEXT,
                   typeIcon TEXT,
                   rarity TEXT,
                   rarityIcon TEXT,
                   domain_1 TEXT,
                   domainIcon_1 TEXT,
                   domain_2 TEXT,
                   domainIcon_2 TEXT,
                   energy TEXT,
                   might TEXT,
                   mightBonus TEXT,
                   power TEXT,
                   tags TEXT,
                   illustrator TEXT,
                   ability TEXT,
                   effect TEXT,
                   image TEXT
                )
                ''')
    
    conn.commit()
    conn.close()
    print("Datenbank erfolgreich erstellt!")

def insert_cards_into_db(card_data):
    conn = sqlite3.connect('riftbound.db')
    cursor = conn.cursor()

    sql = '''
        INSERT OR REPLACE INTO cards (
            id, name, collectorNumber, publicCode,
            setName, cardType, superType, typeIcon, rarity,
            rarityIcon, domain_1, domainIcon_1, domain_2,
            domainIcon_2, energy, might, mightBonus, 
            power, tags, illustrator, ability,
            effect, image
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
    
    cursor.executemany(sql, card_data)
    conn.commit()
    conn.close()
    print(f"Erfolgreich {len(card_data)} Daten in die Datenbank geschrieben!")

