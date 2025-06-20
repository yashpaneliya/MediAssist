# 🚀 MediAssist API

---

## 🛠️ Features

---

## 📦 Requirements

- Python 3.11+
- Redis (running locally or remotely)
- `pip` / `venv` / Docker (recommended)

---

## 🚀 Local Development

### 1. Clone the repo
```bash
git clone https://github.com/yashpaneliya/mediassist.git
cd mediassist-api
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set environment variables
Create a `.env` file in the root directory with the following content:
```env
APP_NAME=MediAssist
DEBUG=True
API_VERSION=v1
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_CACHE_TTL=3600
```

### 4. Run the application
```bash
uvicorn app.main:app --reload
```

### 5. Access the API
Open your browser and navigate to:
```
http://localhost:8000/
```

Swagger UI should be available at:
```
http://localhost:8000/docs
```

Redoc API documentation at:
```
http://localhost:8000/redoc
```

