from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import pandas as pd

# Загружаем наши данные один раз при старте сервера
df = pd.read_csv("anime_report.csv")

class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Если пользователь зашел на главную страницу
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write("<h1>Добро пожаловать на мой сервер!</h1><p>Зайди на /api/anime, чтобы увидеть данные.</p>".encode('utf-8'))
        
        # Если пользователь запросил API с данными
        elif self.path == '/api/anime':
            self.send_response(200)
            # Говорим браузеру, что отдаем JSON
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            
            # Превращаем таблицу Pandas в JSON-формат
            data_json = df.to_json(orient='records', force_ascii=False)
            
            # Отправляем данные в браузер
            self.wfile.write(data_json.encode('utf-8'))
            
        # Если зашли куда-то еще
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write("<h1>404 - Страница не найдена</h1>".encode('utf-8'))

# Запускаем сервер
port = 8080
server = HTTPServer(('localhost', port), MyHandler)
print(f"Сервер запущен на http://localhost:{port}")
print("Нажми Ctrl+C, чтобы остановить.")

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nСервер остановлен.")