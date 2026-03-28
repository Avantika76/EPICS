# Cattle Breed Detection Backend

FastAPI backend for predicting cattle breeds using a trained model.

---

## 🚀 Setup

1. Clone repo

```
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd Epics_Project_Backend
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Add model folder
   Place `convnext_tiny_best/` inside the project directory.

4. Run server

```
python -m uvicorn app:app --reload
```

5. Open API

```
http://127.0.0.1:8000/docs
```

---

## 📡 Endpoint

**POST /predict**

Upload image → get:

```
{
  "breed": "Jersey",
  "confidence": 0.91
}
```
