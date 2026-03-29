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
    # 1. Gestione Cache: creiamo una chiave basata sugli ingredienti ordinati
    ingredients_list = [i.lower().strip() for i in req.ingredients]
    cache_key = ",".join(sorted(ingredients_list))

    if cache_key in recipe_cache:
        return recipe_cache[cache_key]

    # 2. Definizione del comportamento di sistema (System Role)
    system_instruction = (
        "Sei uno Chef stellato esperto in cucina mediterranea e creativa. "
        "Il tuo compito è generare ricette dettagliate, tecniche e gourmet. "
        "Rispondi ESCLUSIVAMENTE in formato JSON con la chiave 'recipes' contenente un array di oggetti. "
        "Ogni istruzione nel procedimento deve essere esaustiva (minimo 20 parole per passaggio). "
        "Lingua: Italiano."
    )

    # 3. Esempio di Alta Qualità (Few-Shot Example) per guidare lo stile
    example_user_input = "Ingredienti: spaghetti, pomodori pelati, aglio"

    # Questo oggetto serve a mostrare all'IA la precisione richiesta
    example_assistant_output = {
        "recipes": [
            {
                "id": "1",
                "title": "Spaghetti alla Chitarra con Pomodoro San Marzano e Olio all'Aglio Dolce",
                "description": "Un'interpretazione tecnica della tradizione, focalizzata sull'estrazione dei sapori primari e la perfetta emulsione degli amidi.",
                "cook_time": "1 h 10 min",
                "difficulty": "Medium",
                "ingredients": [
                    "Spaghetti alla chitarra 320 g",
                    "Pomodori pelati San Marzano 800 g",
                    "Aglio 1 spicchio (privato dell'anima)",
                    "Olio EVO di alta qualità 30 g",
                    "Basilico fresco in foglie",
                    "Sale marino integrale q.b."
                ],
                "instructions": [
                    "In una padella larga in acciaio, scaldare l'olio EVO con l'aglio privato del germoglio interno; lasciare soffriggere a fiamma bassissima per 5 minuti finché l'aglio non diventa dorato, rilasciando i suoi oli essenziali senza bruciare.",
                    "Aggiungere i pomodori pelati schiacciati a mano, salare e coprire con un coperchio. Cuocere a fuoco lentissimo per circa un'ora, mescolando ogni 15 minuti finché la salsa non si riduce diventando densa, lucida e di un rosso profondo.",
                    "Rimuovere l'aglio e passare il sugo attraverso un passaverdure a fori medi per ottenere una consistenza setosa. Trasferire nuovamente in padella e aggiungere il basilico spezzettato rigorosamente a mano per preservarne gli aromi.",
                    "Cuocere la pasta in abbondante acqua poco salata, scolarla molto al dente e saltarla energicamente nel sugo insieme a un mestolo di acqua di cottura ricca di amido, creando una crema densa che avvolge perfettamente ogni spaghetto."
                ]
            }
        ]
    }

    try:
        # 4. Chiamata API a Groq
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": example_user_input},
                {"role": "assistant", "content": json.dumps(
                    example_assistant_output)},
                {"role": "user", "content": f"Basandoti sulla qualità e lo stile dell'esempio precedente, crea esattamente 4 ricette gourmet usando principalmente: {cache_key}"}
            ],
            response_format={"type": "json_object"}
        )

        # 5. Parsing e validazione risposta
        raw_response = completion.choices[0].message.content
        data = json.loads(raw_response)

        # Estrazione delle ricette (gestisce sia se l'AI mette la chiave 'recipes' sia se risponde con l'array)
        recipes = data.get("recipes", []) if isinstance(data, dict) else data

        # 6. Salvataggio in cache e ritorno dei dati
        recipe_cache[cache_key] = recipes
        return recipes

    except Exception as e:
        print(f"ERRORE GENERAZIONE: {str(e)}")
        # In caso di errore, restituiamo un messaggio chiaro al frontend
        raise HTTPException(
            status_code=500, detail="L'IA ha riscontrato un problema nel cucinare le tue ricette. Riprova tra un istante.")

app.mount("/", StaticFiles(directory="static", html=True), name="static")
