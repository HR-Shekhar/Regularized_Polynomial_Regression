import pickle

with open("PolyFromScratch.pkl", "rb") as f:
    weights = pickle.load(f)

print(weights)
