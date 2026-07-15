from src.prediction.predictor import TrafficSignPredictor

predictor = TrafficSignPredictor()

result = predictor.predict(
    "data/raw/traffic_Data/DATA/0/000_1_0001.png"
)

print("="*50)

print(result)

print("="*50)