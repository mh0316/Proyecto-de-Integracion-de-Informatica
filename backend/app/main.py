import os
import sys
import shutil
import chromadb
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Componentes Core de LlamaIndex
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate

# Proveedores Locales y Conectores
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq

# Librerías para la capa Multimodal (Voz)
import whisper
from gtts import gTTS

# Cargar variables de entorno (.env)
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    print("CRÍTICO: Falta la variable GROQ_API_KEY en el archivo .env para el Chat.")
    sys.exit(1)

# Inicialización de FastAPI
app = FastAPI(
    title="API RAG Multimodal - Chatbot Universitario UFRO",
    description="Backend Corregido - Inicialización de Contexto Persistente y Búsqueda Profunda Adaptativa"
)

# Configuración de CORS amplia
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\n" + "="*50)
print("[*] CONFIGURANDO ARQUITECTURA MULTIMODAL HÍBRIDA RAG (TEXTO + VOZ)")
print("="*50)

# 1. CONFIGURACIÓN DEL MODELO EMBEDDING (LOCAL Y GRATUITO)
print("[+] Inicializando motor de búsqueda local (FastEmbed)...")
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 2. CONFIGURACIÓN DEL LLM CON EL CONECTOR NATIVO DE GROQ (Máxima precisión)
print("[+] Conectando cerebro conversacional (Groq Llama-3.1)...")
Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,            
    max_tokens=512,             
    additional_kwargs={
        "presence_penalty": 0.0,  
        "frequency_penalty": 0.0  
    }
)

# 3. CARGA DEL MODELO DE RECONOCIMIENTO DE VOZ LOCAL (WHISPER)
print("[+] Cargando modelo Whisper para transcripción de audio...")
modelo_whisper = whisper.load_model("tiny")

# 4. DIRECTORIO TEMPORAL DE AUDIO
AUDIO_DIR = "./temp_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# ==========================================================================
# 5. CONEXIÓN A CHROMADB (ALINEADA EXACTAMENTE CON INGEST.PY - SUB-CARPETA)
# ==========================================================================
RUTA_SCRIPT = os.path.dirname(os.path.abspath(__file__)) # backend/app/
RAIZ_BACKEND = os.path.abspath(os.path.join(RUTA_SCRIPT, "..")) # backend/

# Forzamos a que apunte a backend/chroma_db de forma exacta
CHROMA_PATH = os.path.join(RAIZ_BACKEND, "chroma_db") 
COLLECTION = os.getenv("COLLECTION_NAME", "ufro_manuals")

print(f"[*] Conectando a la Base de Datos Física Real en: {CHROMA_PATH}")
cliente_db = chromadb.PersistentClient(path=CHROMA_PATH)
coleccion_chroma = cliente_db.get_or_create_collection(COLLECTION)

# Reconstrucción del almacén vectorial y contexto
storage_vectorial = ChromaVectorStore(chroma_collection=coleccion_chroma)
contexto_almacenamiento = StorageContext.from_defaults(vector_store=storage_vectorial)

# Carga limpia forzada desde el almacenamiento indexado
index = VectorStoreIndex.from_vector_store(
    vector_store=storage_vectorial,
    storage_context=contexto_almacenamiento
)

# ==========================================================================
# 6. DEFINICIÓN DEL SYSTEM PROMPT AVANZADO DE ALTA PRECISIÓN (HIPER-PERFECCIONADO)
# ==========================================================================
PROMPT_UFRO = (
    "|REGLAS DE IDENTIDAD Y ROL|\n"
    "Eres 'Bandurr-IA', el Asistente Virtual Oficial de la Universidad de La Frontera (UFRO). "
    "Tu rol exclusivo es actuar como un orientador institucional automatizado de nivel 1. "
    "Respondes dudas sobre reglamentos de régimen de estudios, convivencia estudiantil y el calendario académico.\n\n"
    
    "|REGLA 1: DETECCIÓN Y TRADUCCIÓN DE LENGUAJE COLOQUIAL CHILENO|\n"
    "Los estudiantes de la UFRO pueden consultar usando modismos o jerga informal chilena. Debes decodificar semánticamente "
    "la intención antes de validar el contexto provisto. Reglas explícitas de equivalencia:\n"
    "- 'Congelar', 'echarse para atrás', 'congelar el año', 'retirarse' -> Se refiere al proceso formal de 'Postergación de estudios' o 'Anulación de semestre'.\n"
    "- 'Echarse un ramo', 'pitiarse un ramo', 'reprobar' -> Se refiere a la 'Reprobación de una asignatura' o pérdida de condición de alumno regular por rendimiento.\n"
    "- 'Ramos', 'materias' -> Se refiere a 'Asignaturas' o 'Actividades curriculares'.\n"
    "- 'Prueba', 'solemne' -> Se refiere a 'Evaluaciones', 'Certámenes' o 'Exámenes'.\n"
    "- 'Dar la cacha', 'apelar' -> Se refiere al proceso de 'Reincorporación' o solicitudes excepcionales ante la Dirección de Carrera o Vicerrectoría.\n"
    "Tu respuesta DEBE redactarse en un tono formal, profesional, empático y claro, usando los términos oficiales de la institución, sin imitar ni repetir los modismos del alumno.\n\n"
    
    "|REGLA 2: FILTRADO DE CONTEXTO ESTRICTO Y CERO ALUCINACIÓN|\n"
    "- Examina minuciosamente el 'Contexto' adjunto. Responde ÚNICAMENTE basándote en los hechos, plazos y artículos que aparezcan de forma explícita allí.\n"
    "- Si el contexto describe situaciones específicas (como causales médicas, financieras, becas o de postparto) y la pregunta del usuario apunta a un trámite general o voluntario, NO asumas que la regla médica aplica para el caso general. Diferencia de forma estricta los requisitos especiales de los ordinarios.\n"
    "- No asumas, no extrapoles y bajo ninguna circunstancia inventes fechas o plazos académicos que no estén estipulados de manera explícita en las líneas provistas.\n\n"
    
    "|REGLA 3: POLÍTICA DE HONESTIDAD RAG COMPLETA|\n"
    "Si la respuesta exacta e inequívoca a la pregunta formulada no se encuentra escrita de forma literal en el contexto provisto abajo, o si la información recuperada es parcial y no responde al foco real de la consulta, debes responder EXACTAMENTE con la siguiente frase de control:\n"
    "'Lo siento, esa información específica no se encuentra en los reglamentos ni en el calendario académico cargado en mi sistema. "
    "Te sugiero consultar directamente con la unidad correspondiente (DDE, DAF o tu Dirección de Carrera).'\n\n"
    
    "|REGLA 4: REQUISITOS DE FORMATO DE SALIDA|\n"
    "- Estructura tus respuestas utilizando párrafos cortos, de lectura ágil y viñetas aclaratorias si hay requisitos.\n"
    "- Sé directo, preciso y claro. Evita introducciones innecesarias como 'Basado en el contexto...' o 'De acuerdo a los reglamentos...'. Comienza directamente respondiendo al núcleo de la consulta.\n\n"
    "------------------------\n"
    "Contexto provisto para la inferencia:\n{context_str}\n"
    "------------------------\n"
    "Pregunta formulada por el alumno: {query_str}\n"
    "Respuesta definitiva de Bandurr-IA:"
)

prompt_ufro_template = PromptTemplate(PROMPT_UFRO)
query_engine = index.as_query_engine(similarity_top_k=5, text_qa_template=prompt_ufro_template)

print("="*50)
print("[¡] CORE RAG + CAPA MULTIMODAL INICIALIZADOS CORRECTAMENTE")
print("="*50 + "\n")

class ConsultaUsuario(BaseModel):
    texto: str

class DatosTTS(BaseModel):
    texto: str

# ==========================================================================
# ENDPOINT PRINCIPAL: PIPELINE TEXT-TO-TEXT RAG (CORREGIDO Y BLINDADO)
# ==========================================================================
@app.post("/api/chat")
async def consultar_chatbot(consulta: ConsultaUsuario):
    if not consulta.texto.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    
    frase_error_control = "no se encuentra en los reglamentos ni en el calendario académico"
    texto_fallback = f"Lo siento, esa información específica {frase_error_control}. Te sugiero consultar directamente con la unidad correspondiente (DDE, DAF o tu Dirección de Carrera)."
    
    try:
        texto_usuario = consulta.texto.lower().strip()
        print(f"[*] Procesando pregunta académica/reglamentaria: '{consulta.texto}'")
        
        query_texto = consulta.texto
        es_consulta_calendario = False
        es_consulta_congelar = False
        
        # ==========================================================================
        # 1. MAPEO DE INTENCIONES ACADÉMICAS (CONGELAR / POSTERGACIÓN / ANULACIÓN)
        # ==========================================================================
        KEYWORDS_CONGELAR = ["congelar", "postergacion", "postergar", "anulacion", "anular", "retirarme", "suspender semestre"]
        es_consulta_congelar = any(palabra in texto_usuario for palabra in KEYWORDS_CONGELAR)
        
        # ==========================================================================
        # 2. OPTIMIZACIÓN CRONOLÓGICA PARA EVITAR CONFUSIÓN DE SEMESTRES
        # ==========================================================================
        KEYWORDS_CALENDARIO = [
            "calendario", "cronograma", "receso", "vacaciones", "feriado", 
            "suspensión de clases", "inicio de clases", "término de clases", 
            "fin de clases", "primer semestre 2026", "segundo semestre 2026",
            "inscripción de ramos", "inscripción de asignaturas"
        ]
        es_consulta_calendario = any(palabra in texto_usuario for palabra in KEYWORDS_CALENDARIO)
        
        # Ejecución secuencial de filtros de reescritura de consultas
        if es_consulta_congelar:
            query_texto = (
                "Conducto regular, plazos y requisitos para la postergación de estudios voluntaria, "
                "anulación de matrícula, renuncia o suspensión temporal del semestre en el Reglamento de Régimen de Estudios UFRO"
            )
            print(f"[+] Query optimizado para Congelar/Postergar RAG: '{query_texto}'")
            
        elif es_consulta_calendario:
            if "segundo semestre" in texto_usuario or "mitad de año" in texto_usuario:
                query_texto = (
                    "Fechas oficiales de mitad de año (meses de julio, agosto, septiembre) del calendario académico UFRO 2026: "
                    "Periodo de inscripción de asignaturas y ramos del segundo semestre del 2026, "
                    "receso estudiantil de invierno, inicio de clases del segundo semestre 2026"
                )
            elif "primer semestre" in texto_usuario or "inicio de año" in texto_usuario:
                query_texto = (
                    "Fechas oficiales de inicio de año (meses de enero, febrero, marzo) del calendario académico UFRO 2026: "
                    "Periodo de inscripción de asignaturas y ramos del primer semestre del 2026, "
                    "matrículas, inicio de clases primer semestre 2026"
                )
            else:
                query_texto = (
                    "Fechas del calendario académico institucional UFRO, periodo oficial de inscripción de asignaturas, "
                    "inscripción de ramos, plazos de matrícula, inicio y término de clases, receso estudiantil de mitad de año, "
                    "primer semestre y segundo semestre del año 2026"
                )
            print(f"[+] Query optimizado con segmentación temporal RAG: '{query_texto}'")

        # Inferencia del motor RAG base (Top_K = 5)
        respuesta_rag = query_engine.query(query_texto)
        texto_respuesta = str(respuesta_rag).strip() if respuesta_rag is not None else ""
        
        # Limpieza de prefijos inyectados por Groq
        texto_respuesta = re.sub(r"^(Respuesta del Soporte Automatizado UFRO:\s*|Respuesta de Bandurr-IA:\s*|Bandurr-IA:\s*)", "", texto_respuesta, flags=re.IGNORECASE).strip()

        # Indicadores de respuestas evasivas o insuficientes que gatillan búsqueda profunda
        indicadores_evasivos = [
            frase_error_control, 
            "no se mencionan sanciones específicas", 
            "no se encuentra información específica",
            "no se detalla",
            "no contiene información"
        ]

        # BÚSQUEDA PROFUNDA ADAPTATIVA (Top_K = 10)
        if (not texto_respuesta or any(ind in texto_respuesta.lower() for ind in indicadores_evasivos)):
            print("[!] Alerta RAG: Respuesta evasiva, vacía o insuficiente en primera instancia.")
            
            if es_consulta_congelar:
                query_profundo = "Artículos completos sobre postergación de estudios voluntaria y anulación de semestre sin causales médicas en el reglamento de régimen de estudios"
                print(f"[+] Activando Búsqueda Profunda para Trámite de Congelamiento (Top_K = 10): '{query_profundo}'")
                
            elif es_consulta_calendario:
                if "segundo semestre" in texto_usuario or "mitad de año" in texto_usuario:
                    query_profundo = "Hitos cronológicos, inicio de clases y fechas de inscripción de ramos del segundo semestre 2026 ufro meses julio agosto"
                else:
                    query_profundo = "Fechas e hitos de suspensiones de clases, receso estudiantil, vacaciones y término del semestre 2026 ufro"
                print(f"[+] Activando Búsqueda Profunda Cronológica (Top_K = 10): '{query_profundo}'")
            else:
                query_profundo = query_texto
                print(f"[+] Activando Búsqueda Profunda Reglamentaria Extendida (Top_K = 10)...")
                
            engine_profundo = index.as_query_engine(similarity_top_k=10, text_qa_template=prompt_ufro_template)
            respuesta_rag = engine_profundo.query(query_profundo)
            texto_respuesta = str(respuesta_rag).strip() if respuesta_rag is not None else ""
            
            # Limpieza de prefijos en la segunda respuesta
            texto_respuesta = re.sub(r"^(Respuesta del Soporte Automatizado UFRO:\s*|Respuesta de Bandurr-IA:\s*|Bandurr-IA:\s*)", "", texto_respuesta, flags=re.IGNORECASE).strip()

        # Escudo defensivo definitivo anti-vacíos
        if not texto_respuesta or len(texto_respuesta.strip()) < 5 or texto_respuesta.lower() in ["none", "null", "empty", ""]:
            texto_respuesta = texto_fallback

        # Inyección dinámica de trazabilidad de archivos fuentes (solo si la respuesta fue exitosa)
        if frase_error_control not in texto_respuesta.lower() and hasattr(respuesta_rag, "source_nodes") and respuesta_rag.source_nodes:
            fuentes_encontradas = set()
            for node in respuesta_rag.source_nodes:
                metadatos = node.node.metadata if hasattr(node.node, "metadata") else {}
                nombre_archivo = metadatos.get("file_name")
                if nombre_archivo:
                    fuentes_encontradas.add(nombre_archivo)
            
            if fuentes_encontradas:
                texto_respuesta += f"\n\nFuente: {', '.join(sorted(fuentes_encontradas))}"
        
        return {
            "respuesta": texto_respuesta,
            "status": "success"
        }
        
    except Exception as e:
        print(f"[!] Error crítico interceptado: {str(e)}")
        return {
            "respuesta": texto_fallback,
            "status": "error"
        }

@app.post("/api/asr")
async def audio_a_texto(file: UploadFile = File(...)):
    try:
        ruta_archivo = os.path.join(AUDIO_DIR, file.filename)
        with open(ruta_archivo, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        resultado = modelo_whisper.transcribe(ruta_archivo, language="es")
        texto_transcrito = resultado.get("text", "").strip()
        if os.path.exists(ruta_archivo): os.remove(ruta_archivo)
        return {"texto": texto_transcrito}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts")
async def texto_a_audio(datos: DatosTTS):
    try:
        texto_limpio = datos.texto.split("Fuente:")[0].replace("*", "").replace("\n", " ").strip()
        ruta_salida = os.path.join(AUDIO_DIR, "respuesta_sintetizada.mp3")
        tts = gTTS(text=texto_limpio, lang="es", tld="cl", slow=False)
        tts.save(ruta_salida)
        return FileResponse(ruta_salida, media_type="audio/mpeg", filename="respuesta.mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def estado_servidor():
    return {"status": "ready"}