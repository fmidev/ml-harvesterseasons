import requests
import pandas as pd
import time

startTime=time.time()
query = (
    'http://ml.harvesterseasons.com:8080/timeseries'
    '?precision=full&timestep=data'
    '&param=utctime,latitude,longitude,SKT-K:CCI:5074:1:0:0'
    '&latlon=60.2280502,24.5964108'
    '&starttime=2002-01-01T00:00:00'
    '&endtime=2002-12-31T00:00:00'
    '&tz=UTC&format=json'
)

response = requests.get(query)
response.raise_for_status()

data_json = response.json()

df = pd.DataFrame(data_json)

df['utctime'] = pd.to_datetime(df['utctime'], format="%Y%m%dT%H%M%S")

print(df.dropna())

print(f"Time taken: {time.time() - startTime} seconds")
