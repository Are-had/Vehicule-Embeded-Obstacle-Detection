import serial
from util import nmea_to_decimal
import config

class GPSModule:
    def __init__(self, port=config.SERIAL_PORT, baud=config.BAUD_RATE):
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            # We add satellite counters here
            self.last_status = {
                "lat": 0.0, 
                "lon": 0.0, 
                "valid": False,
                "detected": 0,
                "used": 0
            }
        except Exception as e:
            print(f"Hardware Error: {e}")

    def update(self):
        """
        Reads the serial port and updates the internal status.
        Call this frequently in your main loop.
        """
        if self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('ascii', errors='replace').strip()
                
                # 1. Update Position and Validity
                if "$GPRMC" in line:
                    data = line.split(',')
                    if len(data) > 6:
                        self.last_status["valid"] = (data[2] == 'A')
                        if self.last_status["valid"]:
                            self.last_status["lat"] = nmea_to_decimal(data[3], data[4])
                            self.last_status["lon"] = nmea_to_decimal(data[5], data[6])

                # 2. Update Satellites Detected
                elif "$GPGSV" in line:
                    parts = line.split(',')
                    if len(parts) > 3:
                        self.last_status["detected"] = int(parts[3])

                # 3. Update Satellites Used
                elif "$GPGGA" in line:
                    parts = line.split(',')
                    if len(parts) > 7 and parts[7] != "":
                        self.last_status["used"] = int(parts[7])
            except:
                pass # Ignore malformed serial lines
                
        return self.last_status
