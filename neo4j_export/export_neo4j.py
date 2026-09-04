"""Esporta l'intero grafo Neo4j (nodi + relazioni) in CSV separati per label/tipo
e in un unico graph.json completo — per consegnare i dati senza dover condividere
un dump binario di Neo4j o rifare l'ingestion da zero.

Uso:
    uv run python neo4j_export/export_neo4j.py

Output:
    neo4j_export/csv/film.csv, actor.csv, director.csv, genre.csv,
                     acted_in.csv, directed.csv, has_genre.csv
    neo4j_export/graph.json
"""

import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "csv"
JSON_PATH = BASE_DIR / "graph.json"

NODE_LABELS = {
    "Film": ["vector_id", "titolo", "anno", "trama"],
    "Actor": ["nome", "tmdb_id", "biografia", "data_nascita", "luogo_nascita"],
    "Director": ["nome", "tmdb_id", "biografia", "data_nascita", "luogo_nascita"],
    "Genre": ["nome"],
}

# (tipo_relazione, label_start, label_end, colonna_id_start, colonna_id_end)
RELATIONSHIPS = [
    ("ACTED_IN", "Actor", "Film", "nome", "vector_id"),
    ("DIRECTED", "Director", "Film", "nome", "vector_id"),
    ("HAS_GENRE", "Film", "Genre", "vector_id", "nome"),
]


def _driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password")),
    )


def export_nodes(driver, graph: dict) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    graph["nodes"] = {}

    for label, props in NODE_LABELS.items():
        csv_path = CSV_DIR / f"{label.lower()}.csv"
        with driver.session() as session, csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=props)
            writer.writeheader()
            result = session.run(f"MATCH (n:{label}) RETURN n")
            rows = []
            for record in result:
                node = dict(record["n"])
                row = {p: node.get(p) for p in props}
                writer.writerow(row)
                rows.append(row)
            graph["nodes"][label] = rows
        print(f"  {label}: {len(rows)} nodi -> {csv_path.relative_to(BASE_DIR.parent)}")


def export_relationships(driver, graph: dict) -> None:
    graph["relationships"] = {}

    for rel_type, label_a, label_b, id_a, id_b in RELATIONSHIPS:
        col_a = f"{label_a.lower()}_{id_a}"
        col_b = f"{label_b.lower()}_{id_b}"
        csv_path = CSV_DIR / f"{rel_type.lower()}.csv"
        query = f"""
            MATCH (a:{label_a})-[r:{rel_type}]->(b:{label_b})
            RETURN a.{id_a} AS a_id, b.{id_b} AS b_id
        """
        with driver.session() as session, csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[col_a, col_b])
            writer.writeheader()
            rows = []
            for record in session.run(query):
                row = {col_a: record["a_id"], col_b: record["b_id"]}
                writer.writerow(row)
                rows.append(row)
            graph["relationships"][rel_type] = rows
        print(f"  {rel_type}: {len(rows)} relazioni -> {csv_path.relative_to(BASE_DIR.parent)}")


def main() -> None:
    driver = _driver()
    driver.verify_connectivity()
    print("Connesso a Neo4j. Esportazione in corso...\n")

    graph: dict = {}
    print("Nodi:")
    export_nodes(driver, graph)
    print("\nRelazioni:")
    export_relationships(driver, graph)

    driver.close()

    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"\nDump completo JSON -> {JSON_PATH.relative_to(BASE_DIR.parent)}")


if __name__ == "__main__":
    main()
