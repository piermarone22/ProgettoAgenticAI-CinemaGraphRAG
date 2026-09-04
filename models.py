from pydantic import BaseModel
from typing import List

class MovieInfo(BaseModel):
    titolo: str
    anno: int
    trama: str
    genere: List[str]
    regista: str
    attori_principali: List[str]

class PersonBio(BaseModel):
    nome: str
    ruolo: str # 'Actor' o 'Director'
    biografia: str