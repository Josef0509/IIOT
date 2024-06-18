import sqlite3
import json
import os


class DB:
    def __init__(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../IIOT_DB.db')
        self.conn = sqlite3.connect(filename)
        self.c = self.conn.cursor()

    def query(self, query: str, params: tuple = None):
        with self.conn:
            if params:
                self.c.execute(query, params)
            else:
                self.c.execute(query)
            answer = self.c.fetchall()
        return answer
        
    def __del__(self):
        self.conn.close()

    def fill_DB(self, message: str):
        result = None

        # receiving: {"dispenser": "red","bottle": "24108", "time" : 1718355204, "fill_level_grams" : 118.12191359541822, "recipe" : 21}
        print("receiving: " + message)
        data = json.loads(message)

        first_key = list(data.keys())[0]
        # print("first_key: " + first_key)

        if first_key == "dispenser":
            color = data["dispenser"]
            bottle = data["bottle"]
            time = data["time"]

            if data.get("vibration-index") is None:
                fill_level_grams = data["fill_level_grams"]
                recipe = data["recipe"]
                result = self.query("INSERT INTO Dispenser (color, bottle, time, fill_level_grams, recipe) VALUES (?,?,?,?,?)", (color, bottle, time, fill_level_grams, recipe))
            else:
                vibration_index = data["vibration-index"]
                result = self.query("INSERT INTO DispVibration (color, bottle, time, vibration_index) VALUES (?,?,?,?)", (color, bottle, time, vibration_index))
        
        elif data.get("temperature_C") is not None:
            temperature_C = data["temperature_C"]
            time = data["time"]
            result = self.query("INSERT INTO Temperature (time, temperature_C) VALUES (?,?)", (time, temperature_C))

        elif data.get("final_weight") is not None:
            bottle = data["bottle"]
            time = data["time"]
            final_weight = data["final_weight"]
            result = self.query("INSERT INTO finalWeight (bottle, time, final_weight) VALUES (?,?,?)", (bottle, time, final_weight))
        
        elif data.get("drop_vibration") is not None:
            bottle = data["bottle"]
            drop_vibrations = data["drop_vibration"]

            i = 0
            for drop_vibration in drop_vibrations:
                print("drop_vibration " + str(i) +": " + str(drop_vibration))
                result = self.query("INSERT INTO dropVibration (bottle, n, dropVibration) VALUES (?,?,?)", (bottle, i, drop_vibration))
                i += 1

        elif data.get("is_cracked") is not None:
            bottle = data["bottle"]
            is_cracked = data["is_cracked"]
            result = self.query("INSERT INTO ground_truth (bottle, is_cracked) VALUES (?,?)", (bottle, is_cracked))
            

        if result is not None:
            success = str(result) == "[]"
            print("Stored success: " + str(success))
        else:
            print("no case for this data: " + str(data))