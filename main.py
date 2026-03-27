import os
import json
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Cache semplice in memoria (per produzione usa Redis o un DB)
recipe_cache = {}


class IngredientRequest(BaseModel):
    ingredients: List[str]


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/suggest-recipes")
async def suggest_recipes(req: IngredientRequest):
    # Creiamo una chiave univoca per la cache basata sugli ingredienti ordinati
    cache_key = ",".join(sorted([i.lower().strip() for i in req.ingredients]))

    if cache_key in recipe_cache:
        return recipe_cache[cache_key]

    prompt = f"""
    Crea esattamente 4 ricette diverse usando principalmente questi ingredienti: {cache_key}.
    Rispondi ESCLUSIVAMENTE con un array JSON di oggetti.
    Ogni oggetto deve avere: id (string), title, description, cook_time (es. "20 mins"),
    difficulty (Easy, Medium, Hard), ingredients (array di stringhe con dosi), instructions (array di stringhe).
    Lingua: Italiano.
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}  # Forza l'output JSON
        )

        # Groq restituisce un oggetto, estraiamo il contenuto
        raw_response = completion.choices[0].message.content
        data = json.loads(raw_response)

        # Se l'AI ha messo le ricette dentro una chiave (molto comune), le estraiamo
        if isinstance(data, dict) and "recipes" in data:
            recipes = data["recipes"]
        elif isinstance(data, dict):
            # Se è un dizionario ma non sappiamo la chiave, cerchiamo la prima lista disponibile
            recipes = next(
                (v for v in data.values() if isinstance(v, list)), [])
        else:
            recipes = data  # Era già una lista

        recipe_cache[cache_key] = recipes
        return recipes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/", StaticFiles(directory="static", html=True), name="static")
