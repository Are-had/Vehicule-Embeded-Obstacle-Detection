def format_filename(side, timestamp, lat, lon):
    # On remplace les points par des tirets pour éviter les erreurs de fichier
    # Exemple: frame_G_143005_48-847_2-358.jpg
    s_lat = str(lat).replace('.', '-')
    s_lon = str(lon).replace('.', '-')
    return f"frame_{side}_{timestamp}_{s_lat}_{s_lon}.jpg"


def nmea_to_decimal(value, direction):
    # 1. Sécurisation : si la valeur est vide, on retourne 0.0
    if not value or value == "": return 0.0
    
    # 2. On cherche le point décimal dans la chaîne (ex: 4850.85355)
    dot = value.find('.')
    
    # 3. Le "Secret" : Dans DDMM.MMMM, les minutes occupent TOUJOURS 
    # les 2 chiffres avant le point. On coupe donc la chaîne 2 caractères avant le point.
    deg = float(value[:dot-2])      # Résultat : 48.0
    minutes = float(value[dot-2:])  # Résultat : 50.85355
    
    # 4. Conversion : 1 degré = 60 minutes
    decimal = deg + (minutes / 60.0)
    
    # 5. Gestion des hémisphères
    # Nord (N) et Est (E) sont positifs.
    # Sud (S) et Ouest (W) sont négatifs.
    if direction in ['S', 'W']: 
        decimal = -decimal
        
    return round(decimal, 6) # On garde 6 chiffres après la virgule (précision ~10cm)




def save_metadata(image_path, lat, lon, timestamp, sats_used):
    """
    Creates a CSV file with the same name as the image to store GPS info.
    Example: frame_L_123.jpg -> frame_L_123.csv
    """
    # Replace .jpg with .csv
    csv_path = image_path.rsplit('.', 1)[0] + '.csv'
    
    try:
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["Parameter", "Value"])
            # Data
            writer.writerow(["Timestamp", timestamp])
            writer.writerow(["Latitude", lat])
            writer.writerow(["Longitude", lon])
            writer.writerow(["Satellites_Used", sats_used])
            writer.writerow(["Google_Maps_Link", f"https://www.google.com/maps?q={lat},{lon}"])
        print(f" Metadata saved: {os.path.basename(csv_path)}")
    except Exception as e:
        print(f"Error saving metadata CSV: {e}")
