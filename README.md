# Dokumentation

## `getData.py`

-   Es werden die Topics des MQTT-Broker subscribed.
-   Die empfangenen Daten werden dekodiert.
-   Die dekodierten Daten werden in die SQlite Datenbank gespeichert.

### `config.py`
In diesem File wird die Konfiguration des MQTT-Brokers abgehandelt

## `DB_c.py`

-   In diesem Skript wird eine Klasse "DB" erstellt:

#### Verbindung zur SQlite Datenbank
-   Es wird eine Verbindung zur SQlite Datenbank erstellt und es können Queries durchgeführt werden

#### Füllen der Datenbank
-   Die dekodierten Daten werden mithilfe von SQL-Commands in die Datenbank gespeichert.


## `Datenbank`
-   Als Datenbank wird eine SQlite Datenbank verwendet.


## `Visualisierung`
-   Zur Visualisierung wird plotly verwendet

![placeholder](images\placeholder.png)


