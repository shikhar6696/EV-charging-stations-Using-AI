from flask import Flask, render_template, request
import osmnx as ox
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import folium
import pandas as pd
import os

app = Flask(__name__)

# Path to the uploaded EV stations CSV file
CSV_FILE_PATH = r"C:\\Users\\LENOVO\\Downloads\\ev-charging-stations-india.csv"

# Ensure the static folder for saving maps
if not os.path.exists('static'):
    os.makedirs('static')

# Cache for EV stations data
ev_data_cache = None

# Load the EV stations data from the CSV file
def load_ev_stations_data():
    global ev_data_cache
    if ev_data_cache is None:
        try:
            ev_data = pd.read_csv(CSV_FILE_PATH)
            ev_data.columns = ev_data.columns.str.strip()
            ev_data.rename(columns={'Lattitude': 'Latitude'}, inplace=True)

            # Check for required columns
            if 'Latitude' not in ev_data.columns or 'Longitude' not in ev_data.columns:
                raise KeyError("Required columns 'Latitude' or 'Longitude' are missing.")

            # Convert Latitude and Longitude to numeric
            ev_data['Latitude'] = pd.to_numeric(ev_data['Latitude'], errors='coerce')
            ev_data['Longitude'] = pd.to_numeric(ev_data['Longitude'], errors='coerce')

            # Drop invalid rows
            ev_data.dropna(subset=['Latitude', 'Longitude'], inplace=True)

            ev_data_cache = ev_data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    return ev_data_cache

# Step 1: Get city road network using OSMnx
def get_city_data(city_name):
    try:
        G = ox.graph_from_place(city_name, network_type='drive')
        nodes, edges = ox.graph_to_gdfs(G)
        return nodes, edges
    except Exception as e:
        print("Error fetching city data:", e)
        return None, None

# Step 2: K-Means clustering to find optimal EV station locations
def cluster_ev_stations(locations, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    kmeans.fit(locations)
    return kmeans.cluster_centers_

# Step 3: Simple linear regression to predict EV demand based on population density
def predict_ev_demand(pop_density):
    X = np.array([[10000], [20000], [30000], [40000]])
    y = np.array([10, 20, 35, 50])

    model = LinearRegression()
    model.fit(X, y)

    predicted_demand = model.predict([[pop_density]])
    return predicted_demand

# Step 4: Visualize the station locations on a map using folium
def create_map(existing_stations, predicted_stations):
    india_center = [20.5937, 78.9629]
    city_map = folium.Map(location=india_center, zoom_start=5)

    # Plot existing stations
    for station in existing_stations:
        folium.Marker(
            location=[station[0], station[1]],
            popup="Existing EV Charging Station",
            icon=folium.Icon(color="blue")
        ).add_to(city_map)

    # Plot predicted stations
    for station in predicted_stations:
        folium.Marker(
            location=[station[0], station[1]],
            popup="Predicted EV Charging Station",
            icon=folium.Icon(color="green")
        ).add_to(city_map)

    map_file_path = os.path.join('static', 'ev_charging_map.html')
    city_map.save(map_file_path)

    return map_file_path

# Function to predict the number of stations based on population density
def predict_station_count(pop_density):
    return pop_density // 1000

@app.route('/')
def index():
    ev_data = load_ev_stations_data()

    if ev_data is not None:
        locations = ev_data[['Latitude', 'Longitude']].values
        predicted_stations = cluster_ev_stations(locations, n_clusters=5)
        map_path = create_map(locations, predicted_stations)
        return render_template('index.html', map_path=map_path, predicted_stations=None)
    else:
        return "Error loading EV station data."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pop_density = int(request.form['pop_density'])

        predicted_demand = predict_ev_demand(pop_density)
        predicted_station_count = predict_station_count(pop_density)

        ev_data = load_ev_stations_data()
        if ev_data is not None:
            locations = ev_data[['Latitude', 'Longitude']].values
            predicted_stations = cluster_ev_stations(locations, n_clusters=predicted_station_count)
            map_path = create_map(locations, predicted_stations)
        else:
            map_path = None

        return render_template('index.html', map_path=map_path, predicted_stations=predicted_station_count)

    except ValueError:
        return "Invalid input for population density. Please enter a valid number."

if __name__ == '__main__':
    app.run(debug=True)
