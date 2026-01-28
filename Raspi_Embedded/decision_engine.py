# decision_engine.py

ALLOWED_FILES = [
    "02_Hanns_Klemm_Str_44_000000_000020_leftImg8bit.jpg",
    "04_Maurener_Weg_8_000003_000020_leftImg8bit.jpg",
    "02_Hanns_Klemm_Str_44_000003_000040_leftImg8bit.jpg"
]

def should_send_data(frame_id):
    if frame_id in ALLOWED_FILES:
        print("Frame " + frame_id + " - Will be sent")
        return True
    else:
        print("Frame " + frame_id + " - Skip")
        return False
