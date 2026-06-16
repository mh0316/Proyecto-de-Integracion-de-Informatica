import os
import sys
import shutil
import chromadb
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.readers.file import PDFReader

load_dotenv()

def iniciar_ingesta():
    print("\n" + "="*50)
    print(" PIPELINE DE INGESTA RAG MULTIFORMATO - UFRO 2026")
    print("="*50)

    # 1. RESOLUCIÓN DE RUTAS DEL PROYECTO
    RUTA_SCRIPT = os.path.dirname(os.path.abspath(__file__)) 
    
    # Subimos 3 niveles para llegar a la raíz general donde está la carpeta 'data'
    RAIZ_PROYECTO_GENERAL = os.path.abspath(os.path.join(RUTA_SCRIPT, "..", "..", "..")) 
    DATA_DIR = os.path.join(RAIZ_PROYECTO_GENERAL, "data")
    
    # Subimos 2 niveles para llegar a la raíz de 'backend' donde vive 'chroma_db'
    RAIZ_BACKEND = os.path.abspath(os.path.join(RUTA_SCRIPT, "..", ".."))
    CHROMA_PATH = os.path.join(RAIZ_BACKEND, os.getenv("CHROMA_DB_PATH", "chroma_db"))
    COLLECTION = os.getenv("COLLECTION_NAME", "ufro_manuals")

    print(f"[*] Carpeta de origen detectada: {DATA_DIR}")
    print(f"[*] Base de datos destino: {CHROMA_PATH}")

    # 2. VALIDACIÓN DE ENTRADA
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"\n[!] Error Crítico: La carpeta '{DATA_DIR}' no existe o está vacía.")
        return

    # 3. LIMPIEZA AUTOMÁTICA DE CHROMADB (Evita duplicados y conflictos de IDs)
    if os.path.exists(CHROMA_PATH):
        print(f"[*] Detectada base de datos previa. Limpiando para re-indexación limpia...")
        try:
            shutil.rmtree(CHROMA_PATH)
            print("[+] Base de datos antigua eliminada con éxito.")
        except Exception as e:
            print(f"[!] Advertencia al limpiar directorio: {e}")

    # 4. CONFIGURACIÓN EXPLICITA DE LECTORES (NATIVOS Y AUTOMÁTICOS)
    extensiones_admitidas = [".pdf", ".txt"]

    print(f"[*] Extrayendo información de los archivos en data/...")
    lector = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        required_exts=extensiones_admitidas,
        recursive=True
    )

    documentos = lector.load_data()
    print(f"[+] Éxito: Se procesaron {len(documentos)} elementos/páginas de información.")

    # 5. CONFIGURACIÓN DE HIPERPARÁMETROS LOCALES (FastEmbed)
    print("[*] Cargando motor de embeddings local (FastEmbed)...")
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = None  
    Settings.text_splitter = TokenTextSplitter(chunk_size=1200, chunk_overlap=300)

    # 6. INICIALIZACIÓN DE ENTORNO VECTORIAL PERSISTENTE
    cliente_db = chromadb.PersistentClient(path=CHROMA_PATH)
    coleccion_chroma = cliente_db.get_or_create_collection(COLLECTION)

    storage_vectorial = ChromaVectorStore(chroma_collection=coleccion_chroma)
    contexto_almacenamiento = StorageContext.from_defaults(vector_store=storage_vectorial)

    # 7. GENERACIÓN DE EMBEDDINGS E INDEXACIÓN
    print("[*] Generando vectores matemáticos de forma local en CPU (Gratis)...")
    index = VectorStoreIndex.from_documents(
        documentos,
        storage_context=contexto_almacenamiento,
        show_progress=True
    )

    print("="*50)
    print("¡PROCESO DE INGESTA COMPLETADO CON ÉXITO!")
    print(f"-> Base de datos actualizada en: {CHROMA_PATH}")
    print("="*50 + "\n")

if __name__ == "__main__":
    iniciar_ingesta()