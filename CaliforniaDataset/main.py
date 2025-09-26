import json
import csv

# https://mcauleylab.ucsd.edu:8443/public_datasets/gdrive/googlelocal/

input_file = 'meta-California.json'
output_file = 'panaderias_meta_california.csv'

fields = [
    'name', 'address', 'latitude', 'longitude', 'category', 'avg_rating', 'num_of_reviews', 'price', 'state', 'url'
]

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', newline='', encoding='utf-8') as fout:
    writer = csv.DictWriter(fout, fieldnames=fields)
    writer.writeheader()
    for line in fin:
        try:
            place = json.loads(line)
            # Buscar "bakery" en cualquier categoría (ignorando mayúsculas/minúsculas)
            categories = [c.lower() for c in place.get('category', [])]
            if any('bakery' in c or 'backery' in c for c in categories):
                writer.writerow({
                    'name': place.get('name'),
                    'address': place.get('address'),
                    'latitude': place.get('latitude'),
                    'longitude': place.get('longitude'),
                    'category': ', '.join(place.get('category', [])),
                    'avg_rating': place.get('avg_rating'),
                    'num_of_reviews': place.get('num_of_reviews'),
                    'price': place.get('price'),
                    'state': place.get('state'),
                    'url': place.get('url')
                })
        except Exception as e:
            continue  # Salta líneas mal formateadas

print("¡CSV generado con éxito!")
