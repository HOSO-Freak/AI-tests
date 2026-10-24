import pandas as pd
import json
from sklearn.linear_model import LinearRegression
import numpy as np

with open("anime_clean.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Очищаем данные: удаляем строки, где нет эпизодов или рейтинга
df = df.dropna(subset=["episodes", "score", "year"])

# ТЕПЕРЬ ИСПОЛЬЗУЕМ ДВА ПРИЗНАКА: эпизоды и год
X = df[["episodes", "year"]] 
y = df["score"]

model = LinearRegression()
model.fit(X, y)

# Предсказываем для 100 эпизодов, вышедших в 2020 году
new_anime = [[100, 2020]] 
predicted_rating = model.predict(new_anime)
print(f"Для 100 эпизодов (2020 год) модель предсказывает: {predicted_rating[0]:.2f}")

# Наруто: 220 эпизодов, 2002 год
naruto_data = [[220, 2002]]
naruto_pred = model.predict(naruto_data)
print(f"Для Наруто (220 эп. 2002 г.) модель предсказывает: {naruto_pred[0]:.2f}")
print(f"А реальный рейтинг Наруто: {df[df['name'] == 'Naruto']['score'].values[0]:.2f}")