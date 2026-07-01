import requests
from bs4 import BeautifulSoup
import json
# Own Modules
from src.db.database import init_db, insert_cards_into_db, extract_id_image

website = 'https://riftbound.leagueoflegends.com/de-de/card-gallery/'

response = requests.get(website)

print(f"Status Code: {response.status_code}")
#print(f"HTML Text: {response.text}")

soup = BeautifulSoup(response.text, 'html.parser')

# Search after <script> Tag and '__NEXT_DATA__' ID as common in Next.js
id_tag = soup.find('script', id='__NEXT_DATA__')

all_data = json.loads(id_tag.string)
# props -> pageProps -> page -> blades -> 3tes Blade -> cards -> items
blades = all_data.get('props', {}).get('pageProps', {}).get('page', {}).get('blades', {})
galery_blade = blades[2] # in diesem Blade ist die Kartengalerie
card_list = galery_blade.get('cards', {}).get('items', [])

print(f"Anzahl Karten in card_list: {len(card_list)}")

# -----Extrakt Card Data-----
card_data = []

for card in card_list:
    id = card.get('id')
    name = card.get('name')
    collectorNumber = card.get('collectorNumber')
    publicCode = card.get('publicCode')
    setName = card.get('set', {}).get('value', {}).get('label')
    if card.get('cardType', {}).get('type', []):
        cardType = card.get('cardType', {}).get('type', [])[0].get('label')
        typeIcon = card.get('cardType', {}).get('type', [])[0].get('icon', {}).get('url')
    else:
        cardType = None
        typeIcon = None
    if card.get('cardType', {}).get('superType', []):
        superType = card.get('cardType', {}).get('superType', [])[0].get('label')
    else:
        superType = None
    rarity = card.get('rarity', {}).get('value', {}).get('label')
    rarityIcon = card.get('rarity', {}).get('value', {}).get('icon', {}).get('url')
    domain_1 = card.get('domain', {}).get('values', [])[0].get('label')
    domainIcon_1 = card.get('domain', {}).get('values', [])[0].get('icon', {}).get('url')
    if len(card.get('domain', {}).get('values', [])) > 1:
        domain_2 = card.get('domain', {}).get('values', [])[1].get('label')
        domain_icon_2 = card.get('domain', {}).get('values', [])[1].get('icon', {}).get('url')
    else:
        domain_2 = None
        domainIcon_2 = None
    energy = card.get('energy', {}).get('value', {}).get('label')
    might = card.get('might', {}).get('value', {}).get('label')
    mightBonus = card.get('mightBonus', {}).get('value', {}).get('label')           
    power = card.get('power', {}).get('value', {}).get('label')
    tags_list = card.get('tags', {}).get('tags', [])
    tags = ", ".join(tags_list)
    illustrator = card.get('illustrator', {}).get('values', [])[0].get('label')
    ability = card.get('text', {}).get('richText', {}).get('body')
    effect = card.get('effect', {}).get('richText', {}).get('body')
    image = card.get('cardImage', {}).get('url')

    card_tuple = (id, 
                  name, 
                  collectorNumber, 
                  publicCode, 
                  setName, 
                  cardType, 
                  superType,
                  typeIcon, 
                  rarity,
                  rarityIcon,
                  domain_1,
                  domainIcon_1,
                  domain_2,
                  domainIcon_2,
                  energy,
                  might, 
                  mightBonus,
                  power,
                  tags,
                  illustrator,
                  ability,
                  effect,
                  image)
    
    card_data.append(card_tuple)
print(f"{len(card_data)} Kartendaten in Liste übertragen!")
# ----- End -----

# Initialize Database
init_db()

# Write in Database
insert_cards_into_db(card_data)

# ------Card Search (Helper)------
# query_pubcode = 'UNL-001/219'

# for card in card_list:
#     card_pubcode = card.get('publicCode')
#     if query_pubcode.lower() == card_pubcode.lower():
#         print(card)
# ----- End -----