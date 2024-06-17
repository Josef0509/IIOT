import sys
import os
import paho.mqtt.client as mqtt
import configparser

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database import DB_c

# Define the path to your configuration file
config_file_path = 'MQTT\\mqtt_init.config'

# Read configuration file
config = configparser.ConfigParser()
config.read(config_file_path)

# Debug: Print the sections found in the config file
print("Sections found:", config.sections())

# Ensure the sections and keys are present
if 'MQTT' not in config:
    raise KeyError("Section 'MQTT' not found in the configuration file")
if 'TOPICS' not in config:
    raise KeyError("Section 'TOPICS' not found in the configuration file")

broker = config['MQTT'].get('broker')
port = config['MQTT'].getint('port')
username = config['MQTT'].get('username')
password = config['MQTT'].get('password')
topics = config['TOPICS'].get('topics').split(',\n')

payload = "on"
myDB = DB_c.DB()

# create function for callback
def on_message(client, userdata, message):
    myDB.fill_DB(message.payload.decode())
    #print("message received ", str(message.payload.decode("utf-8")))

# create client object
mqttc = mqtt.Client()
mqttc.username_pw_set(username, password)
# assign function to callback
mqttc.on_message = on_message
# establish connection
mqttc.connect(broker, port)

# subscribe
for topic in topics:
    mqttc.subscribe(topic.strip(), qos=0)

# Blocking call that processes network traffic, dispatches callbacks and handles reconnecting.
#mqttc.loop_forever()

# Uncomment the following lines if you need to run it in a non-blocking way
while True:
    mqttc.loop(0.5)
