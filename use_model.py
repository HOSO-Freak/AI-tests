import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Описываем ту же самую архитектуру, что и при обучении (Питон должен знать, как выглядит сеть)
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128) 
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28) 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 2. Создаем пустую модель и загружаем в нее сохраненные "мозги" (веса)
model = SimpleNet()
# Загружаем файл, который мы сохранили (см. шаг 1 ниже)
model.load_state_dict(torch.load("digit_model.pth"))
model.eval() # Переводим в режим тестирования (отключаем обучение)

print("Модель успешно загружена из файла!")

# 3. Загружаем тестовые данные, чтобы проверить, что модель не забыла, как работать
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True) # batch_size=1, чтобы брать по одной картинке

# 4. Берем случайную картинку из теста и просим модель угадать
data, target = next(iter(test_loader)) # Берем один случайный элемент
with torch.no_grad(): # Не считаем градиенты (экономит память)
    output = model(data)
    
    # output - это 10 чисел (шансы для каждой цифры). Нам нужна та, где число больше.
    predicted = torch.argmax(output, dim=1).item() 
    real_answer = target.item()

#Показ изображения
import matplotlib.pyplot as plt
image = data.squeeze().numpy()
plt.imshow(image, cmap='gray')
plt.title(f"Нейросеть думает {predicted}")
plt.show()

print(f"\nСлучайная картинка:")
print(f"Нейросеть думает, что это цифра: {predicted}")
print(f"А на самом деле там: {real_answer}")
if predicted == real_answer:
    print("✅ Угадала!")
else:
    print("❌ Ошиблась.")