import sys
import os
import paho.mqtt.client as mqtt
import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database import DB_c

# Load configuration from config.py
broker = config.BROKER
port = config.PORT
topics = config.TOPICS
username = config.USERNAME
password = config.PASSWORD

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