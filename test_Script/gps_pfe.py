import serial
import pynmea2
import time

# Configuration du port pour le Raspberry Pi 5
port = "/dev/serial0"
baud = 9600

def run_gps_analysis():
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=1)
        print("--- Demarrage du diagnostic GPS ---")
        
        while True:
            start_time = time.time()
            lat, lon, num_sats, fix_type = None, None, 0, 0
            
            # Lecture des donnees pendant un cycle de 2 secondes
            while time.time() < start_time + 2:
                line = ser.readline().decode('ascii', errors='replace').strip()
                
                # Trame GGA : Satellites detectes et coordonnees
                if "GGA" in line:
                    try:
                        msg = pynmea2.parse(line)
                        num_sats = int(msg.num_sats)
                        if msg.gps_qual > 0:
                            lat, lon = msg.latitude, msg.longitude
                    except:
                        continue
                
                # Trame GSA : Satellites utilises pour le calcul
                if "GSA" in line:
                    try:
                        msg = pynmea2.parse(line)
                        fix_type = int(msg.mode_fix_type)
                    except:
                        continue

            # Affichage des resultats
            print("------------------------------")
            print("Temps: " + time.strftime("%H:%M:%S"))
            
            if num_sats == 0:
                print("Statut: No satellite detected")
            else:
                print("Statut: " + str(num_sats) + " satellites detected")

            if fix_type <= 1:
                print("Fix: No satellite used")
            else:
                print("Fix: Satellites used (Fix " + str(fix_type) + "D)")

            if lat and lon:
                print("Position: Lat " + str(round(lat, 6)) + ", Lon " + str(round(lon, 6)))
                # Generation du lien Google Maps
                maps_url = "https://www.google.com/maps?q=" + str(lat) + "," + str(lon)
                print("Google Maps: " + maps_url)
            else:
                print("Position: Searching for signal...")

    except Exception as e:
        print("Erreur: " + str(e))

if __name__ == "__main__":
    run_gps_analysis()
