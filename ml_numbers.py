import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Загрузка данных. 
# Скачиваем базу картинок с цифрами (MNIST) и превращаем их в тензоры (матрицы для нейросети)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. Создаем саму нейросеть (пока простую, 3 слоя)
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Входной слой (28*28 пикселей = 784) -> Скрытый слой (128 нейронов) -> Выходной (10 цифр)
        self.fc1 = nn.Linear(28 * 28, 128) 
        self.relu = nn.ReLU() # Функция активации (добавляет "нелинейность", чтобы сеть была умной)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28) # Разворачиваем картинку в плоскую линию
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 3. Инициализация
model = SimpleNet()
criterion = nn.CrossEntropyLoss() # Функция потерь (считает, насколько сильно сеть ошиблась)
optimizer = optim.Adam(model.parameters(), lr=0.001) # Оптимизатор (тот, кто "крутит ручки" нейронов, чтобы уменьшить ошибку)

# 4. ОБУЧЕНИЕ (самый важный цикл)
print("Начинаем обучение...")
epochs = 12 # Количество раз, которое сеть просмотрит весь датасет
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()        # 1. Обнуляем предыдущие ошибки
        output = model(data)         # 2. Сеть делает предсказание
        loss = criterion(output, target) # 3. Считаем, насколько она ошиблась
        loss.backward()              # 4. Вычисляем, кто именно виноват в ошибке
        optimizer.step()             # 5. Корректируем нейроны
        
    print(f"Эпоха {epoch+1} завершена. Средняя ошибка: {loss.item():.4f}")

# 5. ПРОВЕРКА (тест на данных, которые сеть никогда не видела)
model.eval() # Переводим сеть в режим экзамена
correct = 0
total = 0
with torch.no_grad(): # Не считаем градиенты при тесте (экономит память)
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1) # Выбираем цифру с самым высоким шансом
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"\nТочность на тестовых данных: {100 * correct / total:.2f}%")

# СОХРАНЕНИЕ МОДЕЛИ
torch.save(model.state_dict(), "digit_model.pth")
print("Модель сохранена в digit_model.pth")