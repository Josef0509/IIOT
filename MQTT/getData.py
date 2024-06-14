import sys
import os
import paho.mqtt.client as mqtt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database import DB_c

broker = "158.180.44.197"
port = 1883
topics = ["iot1/teaching_factory_fast/dispenser_red", 
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

"""
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

"""

payload = "on"
myDB = DB_c.DB()

# create function for callback
def on_message(client, userdata, message):
    myDB.fill_DB(message.payload.decode())

# create client object
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.username_pw_set("bobm", "letmein")              
# assign function to callback
mqttc.on_message = on_message                          
# establish connection
mqttc.connect(broker,port)                 
           
# subscribe
for topic in topics:
    mqttc.subscribe(topic, qos=0)

# Blocking call that processes network traffic, dispatches callbacks and handles reconnecting.
#mqttc.loop_forever()

while True:
    mqttc.loop(0.5)