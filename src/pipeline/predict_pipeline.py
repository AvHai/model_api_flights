import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        """
        Make predictions using the trained model and preprocessor
        
        Args:
            features: DataFrame with input features
            
        Returns:
            predictions: Array of predictions (0: Not Delayed, 1: Delayed)
            probabilities: Array of delay probabilities
        """
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # pipeline directory
            ROOT_DIR = os.path.dirname(BASE_DIR)  # project root

            model_path = os.path.join(ROOT_DIR, "model.pkl")
            preprocessor_path = os.path.join(ROOT_DIR, "preprocessor.pkl")
            
            logging.info("Loading model and preprocessor...")
            print("Before Loading")
            
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            
            logging.info("Applying preprocessing...")
            data_scaled = preprocessor.transform(features)
            
            logging.info("Making predictions...")
            preds = model.predict(data_scaled)
            
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(data_scaled)
                delay_probability = proba[:, 1]  
            else:
                delay_probability = None
            
            logging.info("Predictions completed successfully")
            return preds, delay_probability
        
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise CustomException(e, sys)

class FlightData:
    def __init__(self,
                 year: int,
                 month: int,
                 day: int,
                 airline: str,
                 origin_airport: str,
                 destination_airport: str,
                 scheduled_departure: int,
                 scheduled_time: int = None,
                 distance: float = None,
                 flight_number: int = None):
        """
        Initialize flight data for prediction
        
        Args:
            year: Year of flight (e.g., 2015)
            month: Month of flight (1-12)
            day: Day of month (1-31)
            airline: Airline code (e.g., 'AA', 'UA', 'DL')
            origin_airport: Origin airport code (e.g., 'JFK', 'LAX')
            destination_airport: Destination airport code (e.g., 'ORD', 'ATL')
            scheduled_departure: Scheduled departure time in HHMM format (e.g., 1430 for 2:30 PM)
            scheduled_time: Scheduled flight time in minutes (optional)
            distance: Flight distance in miles (optional)
            flight_number: Flight number (optional)
        """
        self.year = year
        self.month = month
        self.day = day
        self.airline = airline
        self.origin_airport = origin_airport
        self.destination_airport = destination_airport
        self.scheduled_departure = scheduled_departure
        self.scheduled_time = scheduled_time
        self.distance = distance
        self.flight_number = flight_number

    def get_part_of_day(self, departure_time):
        """Convert departure time to part of day"""
        if pd.isna(departure_time):
            return 'Unknown'
        try:
            departure_time = int(departure_time)
            if 500 <= departure_time < 1200:
                return 'Morning'
            elif 1200 <= departure_time < 1700:
                return 'Afternoon'
            elif 1700 <= departure_time < 2100:
                return 'Evening'
            else:
                return 'Night'
        except:
            return 'Unknown'

    def get_data_as_data_frame(self):
        """
        Convert flight data to DataFrame format expected by the model
        """
        try:
            custom_data_input_dict = {
                "YEAR": [self.year],
                "MONTH": [self.month],
                "DAY": [self.day],
                "AIRLINE": [self.airline],
                "ORIGIN_AIRPORT": [self.origin_airport],
                "DESTINATION_AIRPORT": [self.destination_airport],
                "SCHEDULED_DEPARTURE": [self.scheduled_departure],
            }

            custom_data_input_dict["FLIGHT_NUMBER"] = [self.flight_number if self.flight_number is not None else 1]
            custom_data_input_dict["SCHEDULED_TIME"] = [self.scheduled_time if self.scheduled_time is not None else 120]
            custom_data_input_dict["DISTANCE"] = [self.distance if self.distance is not None else 500]
            
            
            df = pd.DataFrame(custom_data_input_dict)
            
            try:
                date = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']], errors='coerce')
                df['DAY_OF_WEEK'] = date.dt.dayofweek
            except:
                df['DAY_OF_WEEK'] = 1  
            
            
            df['IS_WEEKEND'] = (df['DAY_OF_WEEK'] >= 5).astype(int)
            
            
            df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
            df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
            
            
            df['PART_OF_DAY'] = df['SCHEDULED_DEPARTURE'].apply(self.get_part_of_day)
            
            
            df['DISTANCE_PER_MINUTE'] = df['DISTANCE'] / (df['SCHEDULED_TIME'] + 1)
            
            
            df['ELAPSED_TIME'] = df['SCHEDULED_TIME']  
            df['AIR_TIME'] = df['SCHEDULED_TIME'] * 0.8  
            df['SCHEDULED_ARRIVAL'] = df['SCHEDULED_DEPARTURE'] + df['SCHEDULED_TIME']  
            
            logging.info(f"Created DataFrame with shape: {df.shape}")
            logging.info(f"Columns: {list(df.columns)}")
            
            return df

        except Exception as e:
            logging.error(f"Error creating DataFrame: {str(e)}")
            raise CustomException(e, sys)

def predict_flight_delay(year, month, day, airline, origin_airport, destination_airport, 
                        scheduled_departure, scheduled_time=None, distance=None):
    """
    Convenience function to predict flight delay
    
    Returns:
        dict: Prediction results with delay status and probability
    """
    try:
        flight_data = FlightData(
            year=year,
            month=month,
            day=day,
            airline=airline,
            origin_airport=origin_airport,
            destination_airport=destination_airport,
            scheduled_departure=scheduled_departure,
            scheduled_time=scheduled_time,
            distance=distance
        )
        
        
        pred_df = flight_data.get_data_as_data_frame()
        
        
        predict_pipeline = PredictPipeline()
        predictions, probabilities = predict_pipeline.predict(pred_df)
        
        
        is_delayed = bool(predictions[0])
        delay_probability = float(probabilities[0]) if probabilities is not None else None
        
        result = {
            'is_delayed': is_delayed,
            'delay_probability': delay_probability,
            'prediction_text': 'DELAYED' if is_delayed else 'ON TIME',
            'confidence': delay_probability if is_delayed else (1 - delay_probability) if delay_probability else None
        }
        
        return result
        
    except Exception as e:
        logging.error(f"Error in predict_flight_delay: {str(e)}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        print("="*60)
        print("FLIGHT DELAY PREDICTION TESTING")
        print("="*60)
        
        print("\nTest Case 1: Morning Flight")
        result1 = predict_flight_delay(
            year=2015,
            month=6,
            day=15,
            airline="AA",
            origin_airport="JFK",
            destination_airport="LAX",
            scheduled_departure=800,  
            scheduled_time=360,       
            distance=2475             
        )
        print(f"Prediction: {result1['prediction_text']}")
        print(f"Delay Probability: {result1['delay_probability']:.3f}")
        
        
        print("\nTest Case 2: Evening Flight")
        result2 = predict_flight_delay(
            year=2015,
            month=12,
            day=25,  # Christmas
            airline="UA",
            origin_airport="ORD",
            destination_airport="SFO",
            scheduled_departure=1900,  
            scheduled_time=240,        
            distance=1846              
        )
        print(f"Prediction: {result2['prediction_text']}")
        print(f"Delay Probability: {result2['delay_probability']:.3f}")
        
        print("\nTest Case 3: Short Flight")
        result3 = predict_flight_delay(
            year=2015,
            month=3,
            day=10,
            airline="DL",
            origin_airport="ATL",
            destination_airport="JFK",
            scheduled_departure=1200, 
            scheduled_time=150,        
            distance=760               
        )
        print(f"Prediction: {result3['prediction_text']}")
        print(f"Delay Probability: {result3['delay_probability']:.3f}")
        
        print("\n" + "="*60)
        print("TESTING COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"Testing failed: {e}")
        import traceback
        traceback.print_exc()
