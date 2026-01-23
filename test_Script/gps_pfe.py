import serial
import pynmea2
import time
import csv
import os

# CHANGEMENT ICI : On utilise le port qui fonctionne avec cat
PORT = "/dev/ttyAMA0"
BAUD = 9600

CSV_FILE = "gps_positions.csv"

def append_to_csv(timestamp, lat, lon, num_sats, fix_type):
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["timestamp", "latitude", "longitude", "num_sats", "fix_type", "google_maps"])

        maps_url = f"https://www.google.com/maps?q={lat},{lon}"
        writer.writerow([timestamp, lat, lon, num_sats, fix_type, maps_url])

def run_gps_analysis():
    try:
        # Ouverture du port série
        ser = serial.Serial(PORT, baudrate=BAUD, timeout=1)
        print(f"--- Diagnostic GPS démarré sur {PORT} ---")

        last_lat, last_lon = None, None

        while True:
            start_time = time.time()
            lat, lon, num_sats, fix_type = None, None, 0, 0

            # Lecture pendant un cycle de 2 secondes
            while time.time() < start_time + 2:
                try:
                    line = ser.readline().decode("ascii", errors="replace").strip()
                    
                    if not line:
                        continue

                    # Trame GGA : Satellites et coordonnées
                    if "GGA" in line:
                        msg = pynmea2.parse(line)
                        num_sats = int(msg.num_sats)
                        if msg.gps_qual > 0:
                            lat, lon = msg.latitude, msg.longitude

                    # Trame GSA : Type de Fix
                    if "GSA" in line:
                        msg = pynmea2.parse(line)
                        fix_type = int(msg.mode_fix_type)
                        
                except Exception:
                    continue

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            print("------------------------------")
            print("Temps:", timestamp)

            if num_sats == 0:
                print("Statut: No satellite detected")
            else:
                print(f"Statut: {num_sats} satellites detected")

            if fix_type <= 1:
                print("Fix: No satellite used")
            else:
                print(f"Fix: Satellites used (Fix {fix_type}D)")

            if lat is not None and lon is not None:
                lat_r = round(lat, 6)
                lon_r = round(lon, 6)

                print(f"Position: Lat {lat_r} , Lon {lon_r}")
                print(f"Google Maps: https://www.google.com/maps?q={lat_r},{lon_r}")

                if (last_lat != lat_r) or (last_lon != lon_r):
                    append_to_csv(timestamp, lat_r, lon_r, num_sats, fix_type)
                    print(f"Ecrit dans {CSV_FILE}")
                    last_lat, last_lon = lat_r, lon_r
                else:
                    print("Même position, pas d'écriture CSV")
            else:
                print("Position: Searching for signal...")

    except Exception as e:
        print("Erreur critique:", str(e))

if __name__ == "__main__":
    run_gps_analysis()
