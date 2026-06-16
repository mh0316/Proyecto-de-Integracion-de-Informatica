document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const chatMessages = document.getElementById("chatMessages");
    const typingIndicator = document.getElementById("typingIndicator");
    const sendBtn = document.getElementById("sendBtn");
    
    // Nuevos elementos del DOM para control multimedia
    const audioControlsContainer = document.getElementById("audioControlsContainer");
    const audioAnimation = document.getElementById("audioAnimation");
    const deleteAudioBtn = document.getElementById("deleteAudioBtn");
    const pauseAudioBtn = document.getElementById("pauseAudioBtn");
    const pauseIcon = document.getElementById("pauseIcon");

    const BASE_URL = "http://127.0.0.1:8000/api";
    
    // Variables globales de control multimedia
    let mediaRecorder = null;
    let fragmentosAudio = [];
    let estaGrabando = false;
    let seDebeEnviar = true; // Flag para validar si el audio se procesa o se destruye
    let audioActual = null;  // Control de ECO de respuestas del bot

    // 1. ESCUCHADOR DE ENTRADA EN TIEMPO REAL (TEXTO)
    userInput.addEventListener("input", () => {
        if (estaGrabando) return; 
        if (userInput.value.trim().length > 0) {
            sendBtn.className = "mode-send";
        } else {
            sendBtn.className = "mode-mic";
        }
    });

    // 2. INTERCEPTOR DEL CLICK EN EL BOTÓN PRINCIPAL
    sendBtn.addEventListener("click", (e) => {
        // Si está en modo micrófono inicial, disparamos la grabación de voz
        if (sendBtn.classList.contains("mode-mic")) {
            e.preventDefault();
            iniciarGrabacion();
        }
        // Si está en modo envío (mode-send), dejamos que el submit del formulario actúe normalmente
    });

    // 3. BOTÓN SECUNDARIO: BASURERO (DESCARTAR AUDIO)
    deleteAudioBtn.addEventListener("click", () => {
        if (!mediaRecorder) return;
        seDebeEnviar = false; // Activamos bandera de destrucción
        mediaRecorder.stop(); 
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        resetearInterfazAudio("Audio descartado correctamente.");
    });

    // 4. BOTÓN SECUNDARIO: PAUSAR / REANUDAR GRABACIÓN
    pauseAudioBtn.addEventListener("click", () => {
        if (!mediaRecorder) return;

        if (mediaRecorder.state === "recording") {
            mediaRecorder.pause(); // Pausa nativa del hardware
            pauseIcon.className = "fa-solid fa-play"; // Cambia ícono a Play
            pauseAudioBtn.title = "Continuar grabación";
            audioAnimation.classList.add("paused"); // Congela las barras en CSS
            userInput.placeholder = "Grabación en pausa...";
        } else if (mediaRecorder.state === "paused") {
            mediaRecorder.resume(); // Reanuda la captura binaria
            pauseIcon.className = "fa-solid fa-pause"; // Vuelve a ícono Pausa
            pauseAudioBtn.title = "Pausar grabación";
            audioAnimation.classList.remove("paused"); // Reactiva las barras en CSS
            userInput.placeholder = "Grabando audio... Habla ahora.";
        }
    });

    // ==========================================================================
    // NÚCLEO: FLUJO LOGÍSTICO MULTIMEDIA
    // ==========================================================================
    async function iniciarGrabacion() {
        fragmentosAudio = [];
        seDebeEnviar = true; // Aseguramos que inicie habilitado para envío
        
        try {
            const flujoMedios = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(flujoMedios);
            
            mediaRecorder.ondataavailable = (evento) => {
                if (evento.data.size > 0) fragmentosAudio.push(evento.data);
            };

            mediaRecorder.onstop = async () => {
                // Si el usuario presionó el basurero, la bandera será false y salimos sin enviar al RAG
                if (!seDebeEnviar) return;

                const blobAudio = new Blob(fragmentosAudio, { type: "audio/wav" });
                resetearInterfazAudio();
                await procesarAudioRAG(blobAudio);
            };

            mediaRecorder.start();
            estaGrabando = true;
            
            // Requerimiento 2: Cambiar instantáneamente a icono de avión de papel (mode-send)
            sendBtn.className = "mode-send"; 
            
            // Desplegar panel superior de utilidades de audio
            audioControlsContainer.style.display = "flex";
            audioAnimation.classList.remove("paused");
            pauseIcon.className = "fa-solid fa-pause";
            
            userInput.disabled = true;
            userInput.removeAttribute("required"); // Quitamos validación nativa para permitir enviar el audio sin texto
            userInput.placeholder = "Grabando audio... Habla ahora.";

        } catch (err) {
            console.error("Error micrófono:", err);
            alert("Otorga permisos de micrófono en la barra del navegador.");
        }
    }

    function resetearInterfazAudio(placeholderTexto = "Escribe tu consulta...") {
        estaGrabando = false;
        userInput.disabled = false;
        userInput.setAttribute("required", "true");
        userInput.placeholder = placeholderTexto;
        userInput.value = "";
        sendBtn.className = "mode-mic";
        audioControlsContainer.style.display = "none";
    }

    // FORMULARIO PRINCIPAL (ENVÍA TANTO TEXTO COMO FIN DE AUDIO)
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        // CASO 1: Si se envía estando en modo grabación de audio activa (o en pausa)
        if (estaGrabando) {
            if (!mediaRecorder) return;
            seDebeEnviar = true; // Confirmamos que el audio debe ir al backend
            mediaRecorder.stop(); // Detiene hardware y dispara de inmediato el onstop() de arriba
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            return;
        }

        // CASO 2: Envío tradicional de texto escrito en teclado
        const mensajeTexto = userInput.value.trim();
        if (!mensajeTexto) return;

        agregarMensaje(mensajeTexto, "user");
        userInput.value = ""; 
        sendBtn.className = "mode-mic";

        await consultarPipelineRAG(mensajeTexto);
    });

    // PIPELINE MULTIMODAL AUDIO -> backend
    async function procesarAudioRAG(blobAudio) {
        typingIndicator.style.display = "flex";
        scrollToBottom();

        try {
            const formData = new FormData();
            formData.append("file", blobAudio, "consulta_alumno.wav");

            const resAsr = await fetch(`${BASE_URL}/asr`, { method: "POST", body: formData });
            if (!resAsr.ok) throw new Error("Fallo transcripción Whisper");
            
            const datosAsr = await resAsr.json();
            const textoPregunta = datosAsr.texto;

            if (!textoPregunta || textoPregunta.trim() === "") {
                typingIndicator.style.display = "none";
                agregarMensaje("No logré entender el audio con claridad. Por favor inténtalo de nuevo.", "bot");
                return;
            }

            agregarMensaje(textoPregunta, "user");
            await consultarPipelineRAG(textoPregunta);

        } catch (error) {
            console.error("Error Multimodal:", error);
            typingIndicator.style.display = "none";
            agregarMensaje("Error al procesar el flujo de audio local.", "bot");
        }
    }

    // CONSULTA FINAL AL MOTOR DE CONOCIMIENTO (RAG)
    async function consultarPipelineRAG(textoConsulta) {
        typingIndicator.style.display = "flex";
        scrollToBottom();

        try {
            const respuestaServidor = await fetch(`${BASE_URL}/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ texto: textoConsulta })
            });

            if (!respuestaServidor.ok) throw new Error("Error en servidor");
            
            const datos = await respuestaServidor.json();
            typingIndicator.style.display = "none";
            agregarMensaje(datos.respuesta, "bot", true); 

        } catch (error) {
            console.error("Error RAG:", error);
            typingIndicator.style.display = "none";
            agregarMensaje("Problema de conexión con el servidor RAG.", "bot");
        }
        scrollToBottom();
    }

    // REPRODUCTOR DINÁMICO DE VOZ (TTS) CON CONTROL DE ECO Y DETENCIÓN DEFINITIVA
    async function reproducirVozIA(textoAReplicar, botonIcono, botonStop) {
        // CORRECCIÓN: Si hay un audio global reproduciéndose de otra burbuja anterior, lo destruimos limpiamente
        if (audioActual) {
            audioActual.pause();
            audioActual.currentTime = 0;
            audioActual = null;
            
            // Reestablecemos todos los estados visuales del chat a su estado base
            document.querySelectorAll('.btn-listen i').forEach(i => i.className = "fa-solid fa-volume-high");
            document.querySelectorAll('.btn-stop').forEach(btn => btn.classList.add('hidden'));
        }

        botonIcono.className = "fa-solid fa-spinner fa-spin";
        try {
            const respuestaTts = await fetch(`${BASE_URL}/tts`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ texto: textoAReplicar })
            });

            if (!respuestaTts.ok) throw new Error("Error TTS");

            const blobAudio = await respuestaTts.blob();
            const urlAudio = URL.createObjectURL(blobAudio);
            
            audioActual = new Audio(urlAudio);
            
            audioActual.onplay = () => { 
                botonIcono.className = "fa-solid fa-volume-high"; 
                botonStop.classList.remove('hidden'); // Muestra el botón cuadrado de detener
            };

            audioActual.onended = () => {
                botonIcono.className = "fa-solid fa-volume-high";
                botonStop.classList.add('hidden'); // Oculta el botón cuadrado al terminar la pista
                audioActual = null;
            };
            
            await audioActual.play();
        } catch (err) {
            console.error("Error TTS:", err);
            botonIcono.className = "fa-solid fa-volume-xmark";
            botonStop.classList.add('hidden');
            audioActual = null;
        }
    }

    // IMPRESIÓN DINÁMICA DE BURBUJAS ADAPTADA A NUEVO CSS Y BOTÓN DE DETENER
    function agregarMensaje(texto, emisor, incluirAudio = false) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", `${emisor}-message`);

        const contentDiv = document.createElement("div");
        contentDiv.classList.add("message-content");
        contentDiv.textContent = texto;
        messageDiv.appendChild(contentDiv);

        // NUEVO: Inyección modular de la fila de controles TTS debajo del mensaje del bot
        if (incluirAudio && emisor === "bot") {
            const controlsContainer = document.createElement("div");
            controlsContainer.classList.add("tts-audio-controls");

            // Botón de Reproducción / Parlante
            const playBtn = document.createElement("button");
            playBtn.type = "button";
            playBtn.classList.add("tts-btn", "btn-listen");
            playBtn.title = "Escuchar respuesta";
            playBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i>';

            // Botón de Detener Definitivo (Cuadrado)
            const stopBtn = document.createElement("button");
            stopBtn.type = "button";
            stopBtn.classList.add("tts-btn", "btn-stop", "hidden"); // Oculto por defecto mediante .hidden
            stopBtn.title = "Detener audio definitivamente";
            stopBtn.innerHTML = '<i class="fa-solid fa-square"></i>';

            // Escuchador del botón Reproducir/Parlante
            playBtn.addEventListener("click", () => {
                const iconoPlay = playBtn.querySelector("i");
                reproducirVozIA(texto, iconoPlay, stopBtn);
            });

            // Escuchador del nuevo botón Detener (Acción Total y Definitiva)
            stopBtn.addEventListener("click", () => {
                if (audioActual) {
                    audioActual.pause();          // Detiene la vibración física del audio
                    audioActual.currentTime = 0;   // Regresa el puntero al inicio
                    audioActual = null;            // Destruye el objeto para liberar memoria
                }
                playBtn.querySelector("i").className = "fa-solid fa-volume-high"; // Asegura ícono base
                stopBtn.classList.add('hidden'); // Se oculta inmediatamente a sí mismo
            });

            controlsContainer.appendChild(playBtn);
            controlsContainer.appendChild(stopBtn);
            messageDiv.appendChild(controlsContainer);
        }

        const timeSpan = document.createElement("span");
        timeSpan.classList.add("message-time");
        timeSpan.innerText = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        messageDiv.appendChild(timeSpan);
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});