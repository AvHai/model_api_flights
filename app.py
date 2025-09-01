from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime

# ================== Initialize App ==================
app = FastAPI(title="Flight Delay API", version="1.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins, adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
df = pd.read_csv("unique_flights.csv")

# Clean and normalize text columns more thoroughly
df["AIRLINE"] = df["AIRLINE"].astype(str).str.strip().str.replace('"', '').str.upper()
df["ORIGIN_AIRPORT"] = df["ORIGIN_AIRPORT"].astype(str).str.strip().str.replace('"', '').str.upper()
df["DESTINATION_AIRPORT"] = df["DESTINATION_AIRPORT"].astype(str).str.strip().str.replace('"', '').str.upper()

# Ensure delay columns are numeric
df["ARRIVAL_DELAY"] = pd.to_numeric(df["ARRIVAL_DELAY"], errors="coerce")
df["DEPARTURE_DELAY"] = pd.to_numeric(df["DEPARTURE_DELAY"], errors="coerce")

# ================== API 1: Airline Delay Stats ==================
@app.get("/airline-delay-stats")
def airline_delay_stats(airline: str = Query(..., description="Airline code")):
    # Clean input airline code
    airline = airline.strip().upper().replace('"', '')
    
    airline_df = df[df['AIRLINE'] == airline]

    if airline_df.empty:
        # Debug: show available airlines
        available_airlines = df['AIRLINE'].unique()[:10]  # Show first 10
        return JSONResponse(
            content={
                "error": "Airline not found", 
                "requested_airline": airline,
                "available_airlines": available_airlines.tolist()
            }, 
            status_code=404
        )

    total_flights = len(airline_df)
    avg_arrival_delay = airline_df['ARRIVAL_DELAY'].mean()
    avg_departure_delay = airline_df['DEPARTURE_DELAY'].mean()

    # Delay causes
    delay_causes = {}
    for col in ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']:
        if col in airline_df.columns:
            delay_causes[col.lower()] = airline_df[col].mean()

    # Ranking
    airline_group = df.groupby('AIRLINE').agg(
        avg_arrival_delay=('ARRIVAL_DELAY', 'mean'),
        avg_departure_delay=('DEPARTURE_DELAY', 'mean')
    ).reset_index()

    airline_group['rank_by_arrival'] = airline_group['avg_arrival_delay'].rank(method="min")
    airline_group['rank_by_departure'] = airline_group['avg_departure_delay'].rank(method="min")

    this_airline = airline_group[airline_group['AIRLINE'] == airline].iloc[0]

    response = {
        "airline": airline,
        "total_flights": int(total_flights),
        "avg_arrival_delay": round(avg_arrival_delay, 2),
        "avg_departure_delay": round(avg_departure_delay, 2),
        "delays_by_cause": {k: round(v, 2) for k, v in delay_causes.items()},
        "ranking": {
            "rank_by_arrival_delay": int(this_airline['rank_by_arrival']),
            "rank_by_departure_delay": int(this_airline['rank_by_departure']),
            "total_airlines": int(airline_group.shape[0])
        }
    }
    return response


# ================== API 2: Route Performance (by Airline) ==================
@app.get("/route-performance")
def route_performance(
    airline: str = Query(..., description="Airline code (e.g., AA)"),
    origin: str = Query(..., description="Origin airport code (e.g., JFK)"),
    destination: str = Query(..., description="Destination airport code (e.g., LAX)")
):
    # Normalize inputs and clean them
    airline = airline.strip().upper().replace('"', '')
    origin = origin.strip().upper().replace('"', '')
    destination = destination.strip().upper().replace('"', '')

    # Filter airline
    airline_data = df[df["AIRLINE"] == airline]
    if airline_data.empty:
        return JSONResponse(content={"error": f"No flights found for airline {airline}"}, status_code=404)

    # Filter route
    route_data = airline_data[
        (airline_data["ORIGIN_AIRPORT"] == origin) & 
        (airline_data["DESTINATION_AIRPORT"] == destination)
    ]

    if route_data.empty:
        return JSONResponse(
            content={"error": f"No data found for {airline} on route {origin} -> {destination}"},
            status_code=404
        )

    # Get arrival delay data and remove NaN values
    arrival_delays = route_data["ARRIVAL_DELAY"].dropna()
    
    # Compute delay distribution
    delay_0_15 = ((arrival_delays >= 0) & (arrival_delays <= 15)).sum()
    delay_15_60 = ((arrival_delays > 15) & (arrival_delays <= 60)).sum()
    delay_60_plus = (arrival_delays > 60).sum()
    
    # Compute stats safely (ignore NaN)
    route_stats = {
        "airline": airline,
        "origin": origin,
        "destination": destination,
        "total_flights": int(len(route_data)),
        "avg_arrival_delay": round(route_data["ARRIVAL_DELAY"].mean(skipna=True), 2),
        "avg_departure_delay": round(route_data["DEPARTURE_DELAY"].mean(skipna=True), 2),
        "delay_distribution": {
            "0-15min": int(delay_0_15),
            "15-60min": int(delay_15_60),
            "60+min": int(delay_60_plus)
        }
    }

    return route_stats


# ================== NEW API 3: Get Flights by Route with Delay Risk ==================
@app.get("/get_flights_by_route")
def get_flights_by_route(
    origin_airport: str = Query(..., description="Origin airport code (e.g., JFK)"),
    destination_airport: str = Query(..., description="Destination airport code (e.g., LAX)"),
    date: str = Query(None, description="Date in YYYY-MM-DD format (e.g., 2015-01-01) - for reference only")
):
    # Clean inputs
    origin = origin_airport.strip().upper().replace('"', '')
    destination = destination_airport.strip().upper().replace('"', '')
    
    # Date is optional and just for reference since your dataset doesn't have date filtering
    request_date = date if date else "Not specified"
    
    # Filter flights by route
    route_flights = df[
        (df["ORIGIN_AIRPORT"] == origin) & 
        (df["DESTINATION_AIRPORT"] == destination)
    ]
    
    if route_flights.empty:
        return JSONResponse(
            content={"error": f"No flights found for route {origin} -> {destination}"},
            status_code=404
        )
    
    # Get unique flight numbers on this route
    unique_flights = route_flights.groupby(['AIRLINE', 'FLIGHT_NUMBER']).agg({
        'ARRIVAL_DELAY': ['mean', 'std', 'count'],
        'DEPARTURE_DELAY': ['mean', 'std', 'count'],
        'SCHEDULED_DEPARTURE': 'first',
        'SCHEDULED_ARRIVAL': 'first'
    }).reset_index()
    
    # Flatten column names
    unique_flights.columns = [
        'AIRLINE', 'FLIGHT_NUMBER', 
        'avg_arrival_delay', 'std_arrival_delay', 'arrival_delay_count',
        'avg_departure_delay', 'std_departure_delay', 'departure_delay_count',
        'scheduled_departure', 'scheduled_arrival'
    ]
    
    # Calculate delay risk (higher mean delay + higher std = higher risk)
    unique_flights['delay_risk_score'] = (
        unique_flights['avg_arrival_delay'].fillna(0) + 
        unique_flights['std_arrival_delay'].fillna(0)
    )
    
    # Categorize delay risk
    def get_delay_risk_category(score):
        if score <= 0:
            return "Low"
        elif score <= 15:
            return "Medium"
        elif score <= 30:
            return "High"
        else:
            return "Very High"
    
    unique_flights['delay_risk'] = unique_flights['delay_risk_score'].apply(get_delay_risk_category)
    
    # Prepare response
    flights_list = []
    for _, flight in unique_flights.iterrows():
        flights_list.append({
            "airline": flight['AIRLINE'],
            "flight_number": int(flight['FLIGHT_NUMBER']),
            "scheduled_departure": flight['scheduled_departure'],
            "scheduled_arrival": flight['scheduled_arrival'],
            "avg_arrival_delay": round(flight['avg_arrival_delay'], 2) if not pd.isna(flight['avg_arrival_delay']) else None,
            "avg_departure_delay": round(flight['avg_departure_delay'], 2) if not pd.isna(flight['avg_departure_delay']) else None,
            "delay_risk": flight['delay_risk'],
            "delay_risk_score": round(flight['delay_risk_score'], 2),
            "total_historical_flights": int(flight['arrival_delay_count'])
        })
    
    # Sort by delay risk score (ascending - lower risk first)
    flights_list.sort(key=lambda x: x['delay_risk_score'])
    
    response = {
        "route": f"{origin} -> {destination}",
        "requested_date": request_date,
        "total_flights_available": len(flights_list),
        "flights": flights_list
    }
    
    return response


# ================== NEW API 4: Route Performance (All Airlines) ==================
@app.get("/route-performance-all")
def route_performance_all(
    origin: str = Query(..., description="Origin airport code (e.g., JFK)"),
    destination: str = Query(..., description="Destination airport code (e.g., LAX)")
):
    # Clean inputs
    origin = origin.strip().upper().replace('"', '')
    destination = destination.strip().upper().replace('"', '')
    
    # Filter route data (all airlines)
    route_data = df[
        (df["ORIGIN_AIRPORT"] == origin) & 
        (df["DESTINATION_AIRPORT"] == destination)
    ]
    
    if route_data.empty:
        return JSONResponse(
            content={"error": f"No data found for route {origin} -> {destination}"},
            status_code=404
        )
    
    # Overall route statistics
    arrival_delays = route_data["ARRIVAL_DELAY"].dropna()
    departure_delays = route_data["DEPARTURE_DELAY"].dropna()
    
    # Delay distribution
    delay_0_15 = ((arrival_delays >= 0) & (arrival_delays <= 15)).sum()
    delay_15_60 = ((arrival_delays > 15) & (arrival_delays <= 60)).sum()
    delay_60_plus = (arrival_delays > 60).sum()
    
    # Airline breakdown
    airline_stats = route_data.groupby('AIRLINE').agg({
        'ARRIVAL_DELAY': ['mean', 'count'],
        'DEPARTURE_DELAY': 'mean',
        'FLIGHT_NUMBER': 'nunique'
    }).reset_index()
    
    # Flatten column names
    airline_stats.columns = ['airline', 'avg_arrival_delay', 'flight_count', 'avg_departure_delay', 'unique_flights']
    
    # Sort by average arrival delay
    airline_stats = airline_stats.sort_values('avg_arrival_delay')
    
    airlines_list = []
    for _, airline in airline_stats.iterrows():
        airlines_list.append({
            "airline": airline['airline'],
            "avg_arrival_delay": round(airline['avg_arrival_delay'], 2) if not pd.isna(airline['avg_arrival_delay']) else None,
            "avg_departure_delay": round(airline['avg_departure_delay'], 2) if not pd.isna(airline['avg_departure_delay']) else None,
            "total_flights": int(airline['flight_count']),
            "unique_flight_numbers": int(airline['unique_flights'])
        })
    
    # Calculate delay causes (if available)
    delay_causes = {}
    for col in ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']:
        if col in route_data.columns:
            delay_causes[col.lower()] = round(route_data[col].mean(), 2)
    
    route_stats = {
        "route": f"{origin} -> {destination}",
        "total_flights": int(len(route_data)),
        "total_airlines": int(route_data['AIRLINE'].nunique()),
        "avg_arrival_delay": round(route_data["ARRIVAL_DELAY"].mean(skipna=True), 2),
        "avg_departure_delay": round(route_data["DEPARTURE_DELAY"].mean(skipna=True), 2),
        "delay_distribution": {
            "0-15min": int(delay_0_15),
            "15-60min": int(delay_15_60),
            "60+min": int(delay_60_plus)
        },
        "delay_causes": delay_causes,
        "airlines": airlines_list
    }
    
    return route_stats


# ================== Data inspection endpoint ==================
@app.get("/inspect-data")
def inspect_data():
    """Endpoint to inspect the loaded data"""
    return {
        "total_rows": len(df),
        "columns": df.columns.tolist(),
        "unique_airlines": df['AIRLINE'].unique().tolist(),
        "sample_data": df.head(3).to_dict('records'),
        "airline_value_examples": df['AIRLINE'].head(10).tolist()
    }


@app.get("/debug-route")
def debug_route(
    airline: str = Query(..., description="Airline code"),
    origin: str = Query(..., description="Origin airport code"),
    destination: str = Query(..., description="Destination airport code")
):
    """Debug endpoint to see what data is available for a route"""
    # Normalize inputs and clean them
    airline = airline.strip().upper().replace('"', '')
    origin = origin.strip().upper().replace('"', '')
    destination = destination.strip().upper().replace('"', '')
    
    # Check airline
    airline_data = df[df["AIRLINE"] == airline]
    
    # Check route
    route_data = airline_data[
        (airline_data["ORIGIN_AIRPORT"] == origin) & 
        (airline_data["DESTINATION_AIRPORT"] == destination)
    ]
    
    return {
        "airline": airline,
        "origin": origin,
        "destination": destination,
        "airline_exists": not airline_data.empty,
        "route_exists": not route_data.empty,
        "airline_flight_count": len(airline_data),
        "route_flight_count": len(route_data),
        "sample_origins_for_airline": list(airline_data["ORIGIN_AIRPORT"].unique()[:10]) if not airline_data.empty else [],
        "sample_destinations_for_airline": list(airline_data["DESTINATION_AIRPORT"].unique()[:10]) if not airline_data.empty else [],
        "arrival_delay_sample": route_data["ARRIVAL_DELAY"].head().tolist() if not route_data.empty else []
    }


# ================== API 3: ML Prediction ==================
from src.pipeline.predict_pipeline import predict_flight_delay, CustomException

class FlightRequest(BaseModel):
    year: int
    month: int
    day: int
    airline: str
    origin_airport: str
    destination_airport: str
    scheduled_departure: int
    scheduled_time: int = None
    distance: float = None

@app.post("/predict")
def predict_flight(data: FlightRequest):
    try:
        result = predict_flight_delay(
            year=data.year,
            month=data.month,
            day=data.day,
            airline=data.airline,
            origin_airport=data.origin_airport,
            destination_airport=data.destination_airport,
            scheduled_departure=data.scheduled_departure,
            scheduled_time=data.scheduled_time,
            distance=data.distance
        )
        return {"status": "success", "prediction": result}
    except CustomException as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=400)
    
    
    