# other possiilites would be: .ini or .json or .config files, but this is quick and dirty


# MQTT broker configuration
BROKER = "158.180.44.197"
PORT = 1883
USERNAME = "bobm"
PASSWORD = "letmein"

# List of topics to subscribe to
TOPICS = [
    "iot1/teaching_factory_fast/dispenser_red", 
    "iot1/teaching_factory_fast/dispenser_red/vibration", 
    "iot1/teaching_factory_fast/temperature", 
    "iot1/teaching_factory_fast/dispenser_blue",
    "iot1/teaching_factory_fast/dispenser_blue/vibration",
    "iot1/teaching_factory_fast/dispenser_green",
    "iot1/teaching_factory_fast/dispenser_green/vibration",
    "iot1/teaching_factory_fast/scale/final_weight",
    "iot1/teaching_factory_fast/drop_vibration",
    "iot1/teaching_factory_fast/ground_truth"
]
