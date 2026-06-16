import os
from pypdf import PdfReader

def reparar_y_convertir_data():
    # Al estar en la raíz, la carpeta data está al lado del script
    DATA_DIR = "./data"
    
    if not os.path.exists(DATA_DIR):
        print(f"[!] Error: No se encontró la carpeta '{DATA_DIR}' en esta ubicación.")
        return

    print("=" * 65)
    print(" REPARADOR MULTIFORMATO: CONVIRTIENDO BINARIOS A TEXTO PLANO REAL")
    print("=" * 65)

    for nombre_archivo in os.listdir(DATA_DIR):
        ruta_archivo = os.path.join(DATA_DIR, nombre_archivo)
        
        # Ignorar si es un directorio
        if os.path.isdir(ruta_archivo):
            continue

        print(f"\n[*] Analizando archivo: {nombre_archivo}")

        try:
            # Forzamos la lectura como estructura PDF (sirve para .pdf y para el .txt que era un PDF encubierto)
            lector = PdfReader(ruta_archivo)
            texto_extraido = ""

            for i, pagina in enumerate(lector.pages):
                lineas = pagina.extract_text()
                if lineas:
                    texto_extraido += lineas + "\n"

            if texto_extraido.strip():
                # Obtenemos el nombre sin extensión
                nombre_base, extension_original = os.path.splitext(nombre_archivo)
                
                # CASO 1: Si es tu manual de eduroam que vino como .txt binario
                if nombre_archivo == "manual_eduroam.txt":
                    # Sobrescribimos el mismo archivo pero ahora con TEXTO PLANO REAL y UTF-8
                    with open(ruta_archivo, "w", encoding="utf-8") as f:
                        f.write(texto_extraido.strip())
                    print(f"[+] ¡Éxito! El archivo 'manual_eduroam.txt' ha sido saneado a texto plano real.")
                
                # CASO 2: Si son los otros archivos PDFs de reglamentos
                elif extension_original.lower() == ".pdf":
                    # Creamos una copia en .txt para asegurarnos de que LlamaIndex los lea sin problemas
                    ruta_salida_txt = os.path.join(DATA_DIR, f"{nombre_base}.txt")
                    with open(ruta_salida_txt, "w", encoding="utf-8") as f:
                        f.write(texto_extraido.strip())
                    print(f"[+] ¡Éxito! PDF convertido a texto plano en: {ruta_salida_txt}")
                    
                    # Opcional: Eliminamos el PDF viejo para que no queden duplicados en la carpeta data
                    os.remove(ruta_archivo)
                    print(f"[-] Eliminado archivo binario original: {nombre_archivo}")
            else:
                print(f"[!] Advertencia: {nombre_archivo} no contiene texto extraíble (¿Es una imagen/escaneo?).")

        except Exception as e:
            # Si un archivo ya era texto plano real (como un .txt legítimo), PdfReader fallará. 
            # Lo capturamos aquí para no interrumpir el script.
            print(f"[*] El archivo {nombre_archivo} no es un binario de tipo PDF o ya es texto plano legítimo. Saltando... (Detalle: {e})")

    print("\n" + "=" * 65)
    print(" ¡PROCESO DE SANEAMIENTO COMPLETADO!")
    print(" Ahora todos tus archivos en data/ son texto plano (.txt) real.")
    print("=" * 65)

if __name__ == "__main__":
    reparar_y_convertir_data()