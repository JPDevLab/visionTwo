
import io
import base64

import streamlit as st
from PIL import Image, ImageOps
from openai import OpenAI


# ==================================================
# CONFIGURACIÓN
# ==================================================

st.set_page_config(
    page_title="VisionAI | Image Analyzer",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# ESTILOS VISUALES
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #0a1020;
    --panel: #111a2d;
    --border: #293852;
    --text: #f1f5f9;
    --muted: #9aaac0;
    --accent: #818cf8;
}

.stApp {
    background:
        radial-gradient(ellipse at 70% 0%, #20234a 0%, #0a1020 48%);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #0d1425;
    border-right: 1px solid #293852;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.5px;
}

h1 {
    font-size: clamp(2rem, 4vw, 3rem) !important;
    font-weight: 700 !important;
}

p, label {
    color: #cbd5e1;
}

.hero {
    padding: 1.5rem 0 1.2rem;
}

.eyebrow {
    display: inline-block;
    color: #c4b5fd;
    background: #272449;
    border: 1px solid #454078;
    border-radius: 999px;
    padding: 6px 12px;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    margin-bottom: 15px;
}

.hero-subtitle {
    max-width: 700px;
    color: #a5b4cb;
    font-size: 1.03rem;
    line-height: 1.75;
}

.panel {
    background: linear-gradient(145deg, #151f35, #10182a);
    border: 1px solid #2b3b57;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 14px;
}

.panel-title {
    color: #f1f5f9;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.panel-description {
    color: #9eacc1;
    font-size: 0.87rem;
    line-height: 1.6;
}

.section-label {
    color: #a5b4fc;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    margin: 10px 0;
}

.stTextInput input, .stTextArea textarea {
    background: #10192c !important;
    color: #f1f5f9 !important;
    border: 1px solid #34445f !important;
    border-radius: 11px !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 1px #818cf8 !important;
}

[data-testid="stFileUploader"] {
    background: #111b2f;
    border: 1px dashed #526180;
    border-radius: 14px;
    padding: 8px;
}

.stButton > button {
    min-height: 44px;
    border-radius: 11px;
    border: 1px solid #7774e8;
    background: linear-gradient(135deg, #7774e8, #635bcd);
    color: white;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #918cfb, #7774e8);
    border-color: #aaa5ff;
    transform: translateY(-1px);
}

div[data-testid="stMetric"] {
    background: #111b2f;
    border: 1px solid #2b3b57;
    border-radius: 13px;
    padding: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #a5b4cb;
}

div[data-testid="stMetricValue"] {
    color: #f1f5f9;
}

.result-box {
    background: linear-gradient(145deg, #171d38, #111a2d);
    border: 1px solid #494779;
    border-left: 4px solid #918cfb;
    border-radius: 15px;
    padding: 22px;
    margin: 10px 0 20px;
    color: #e2e8f0;
    line-height: 1.8;
}

.sidebar-caption {
    color: #96a6bd;
    font-size: 0.83rem;
    line-height: 1.7;
}

.footer {
    border-top: 1px solid #293852;
    color: #75849b;
    text-align: center;
    font-size: 0.78rem;
    margin-top: 35px;
    padding: 20px 0;
}

hr {
    border-color: #293852;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# ESTADO DE LA APLICACIÓN
# ==================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_signature" not in st.session_state:
    st.session_state.analysis_signature = None


# ==================================================
# BARRA LATERAL
# ==================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 4px 0 18px;">
        <div style="font-size: 2.4rem;">🔎</div>
        <div style="font-family: 'Space Grotesk', sans-serif;
                    font-size: 1.65rem; font-weight: 700;
                    color: #f1f5f9;">VisionAI</div>
        <div style="color: #9aaac0; font-size: 0.82rem;">
            Intelligent Image Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Configuración")

    api_key = st.text_input(
        "Clave API de OpenAI",
        type="password",
        placeholder="sk-...",
        help="Se utiliza para enviar la imagen al modelo de OpenAI.",
    )

    st.caption(
        "Tu clave se utiliza para las solicitudes de esta sesión. "
        "No necesitas guardarla como variable de entorno."
    )

    st.markdown("---")
    st.markdown("### 🧠 Modelo")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">GPT-4o</div>
        <div class="panel-description">
            Análisis visual, descripción de escenas,
            objetos, texto visible y contexto proporcionado.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💡 Consejos")
    st.markdown("""
    <div class="sidebar-caption">
        • Usa imágenes claras y legibles.<br>
        • Añade contexto para orientar el análisis.<br>
        • Formula instrucciones específicas.<br>
        • Comprueba los datos extraídos de gráficos.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑️ Limpiar análisis", use_container_width=True):
        st.session_state.analysis_result = None
        st.session_state.analysis_signature = None
        st.rerun()


# ==================================================
# CABECERA
# ==================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ COMPUTER VISION · GENERATIVE AI</div>
    <h1>Descubre lo que hay en tu imagen.</h1>
    <div class="hero-subtitle">
        Sube una imagen, añade tus instrucciones y deja que la
        inteligencia artificial te ayude a interpretar su contenido.
        Obtén descripciones, identifica elementos y explora detalles.
    </div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# ÁREA PRINCIPAL
# ==================================================

col_image, col_config = st.columns([1.15, 0.85], gap="large")

with col_image:
    st.markdown(
        '<div class="section-label">01 / TU IMAGEN</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png", "webp"],
        help="Formatos admitidos: JPG, JPEG, PNG y WEBP.",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image)
            image.load()

            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True,
            )

            st.caption(
                f"Formato: {image.format or 'Imagen'} · "
                f"Resolución: {image.width} × {image.height} px"
            )

        except Exception:
            st.error("No se pudo abrir la imagen. Prueba con otro archivo.")
            image = None
    else:
        image = None
        st.markdown("""
        <div class="panel" style="text-align:center; padding: 48px 20px;">
            <div style="font-size:3.5rem; margin-bottom:12px;">🖼️</div>
            <div class="panel-title">Tu espacio de análisis</div>
            <div class="panel-description">
                Selecciona una imagen para ver aquí la vista previa.
                <br>JPG · PNG · WEBP
            </div>
        </div>
        """, unsafe_allow_html=True)


with col_config:
    st.markdown(
        '<div class="section-label">02 / INSTRUCCIONES</div>',
        unsafe_allow_html=True,
    )

    analysis_type = st.selectbox(
        "¿Qué deseas analizar?",
        [
            "Descripción general",
            "Identificación de objetos",
            "Lectura de texto (OCR visual)",
            "Análisis de gráfico o tabla",
            "Análisis técnico",
            "Instrucciones personalizadas",
        ],
    )

    prompts = {
        "Descripción general":
            "Describe detalladamente lo que aparece en la imagen. "
            "Organiza la respuesta por elementos principales, "
            "escena y detalles relevantes.",

        "Identificación de objetos":
            "Identifica los objetos y elementos visibles. "
            "Describe su apariencia, ubicación aproximada "
            "y relaciones entre ellos. No inventes objetos.",

        "Lectura de texto (OCR visual)":
            "Transcribe el texto visible en la imagen con la mayor "
            "fidelidad posible. Organiza el resultado por secciones "
            "y señala cualquier parte que no sea legible.",

        "Análisis de gráfico o tabla":
            "Analiza el gráfico o tabla. Explica sus variables, "
            "tendencias, valores y conclusiones que estén respaldadas "
            "por la información visible. No inventes valores ni fechas.",

        "Análisis técnico":
            "Realiza un análisis técnico de la imagen. Describe "
            "sus componentes, características y posibles problemas "
            "visibles. Diferencia claramente las observaciones "
            "de las inferencias.",

        "Instrucciones personalizadas": "",
    }

    custom_prompt = st.text_area(
        "¿Qué quieres saber?",
        value=prompts[analysis_type],
        height=155,
        placeholder="Escribe las instrucciones para la IA...",
    )

    with st.expander("➕ Añadir contexto adicional"):
        additional_details = st.text_area(
            "Contexto opcional",
            placeholder=(
                "Ej.: es un gráfico de consumo eléctrico; "
                "quiero comparar las horas de mayor demanda."
            ),
            height=100,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    analyze_button = st.button(
        "✨ Analizar imagen",
        use_container_width=True,
        type="primary",
    )

    st.caption(
        "El análisis utiliza GPT-4o a través de la API de OpenAI. "
        "El uso de la API puede generar costos."
    )


# ==================================================
# ANÁLISIS
# ==================================================

if analyze_button:
    if image is None or uploaded_file is None:
        st.warning("Primero debes cargar una imagen.")

    elif not api_key.strip():
        st.warning("Ingresa tu clave API en la barra lateral.")

    elif not custom_prompt.strip():
        st.warning("Escribe una instrucción para orientar el análisis.")

    else:
        try:
            # Convertir la imagen a JPEG para mantener
            # un formato compatible con la solicitud.
            rgb_image = image.convert("RGB")
            buffer = io.BytesIO()
            rgb_image.save(buffer, format="JPEG", quality=90)
            base64_image = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

            prompt_text = (
                "Responde siempre en español. "
                "Analiza la imagen basándote en lo que realmente puedes observar. "
                "Distingue los hechos visibles de las inferencias y no inventes "
                "datos que no estén presentes.\n\n"
                f"INSTRUCCIONES:\n{custom_prompt.strip()}"
            )

            if additional_details.strip():
                prompt_text += (
                    "\n\nCONTEXTO ADICIONAL DEL USUARIO:\n"
                    f"{additional_details.strip()}"
                )

            with st.spinner(
                "🔬 Analizando los elementos de la imagen..."
            ):
                client = OpenAI(api_key=api_key.strip())

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt_text,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": (
                                            "data:image/jpeg;base64,"
                                            f"{base64_image}"
                                        )
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=800,
                )

                result = response.choices[0].message.content

                if not result:
                    raise ValueError(
                        "El modelo no devolvió una respuesta."
                    )

                st.session_state.analysis_result = result
                st.session_state.analysis_signature = (
                    uploaded_file.name,
                    len(uploaded_file.getvalue()),
                    analysis_type,
                    custom_prompt,
                    additional_details,
                )

        except Exception as error:
            st.error(
                "No se pudo completar el análisis. "
                "Verifica tu clave API, conexión y acceso al modelo."
            )
            st.caption(f"Detalle técnico: {error}")


# ==================================================
# RESULTADO
# ==================================================

if st.session_state.analysis_result:
    st.markdown("---")

    st.markdown(
        '<div class="section-label">03 / RESULTADO DEL ANÁLISIS</div>',
        unsafe_allow_html=True,
    )

    st.markdown("## ✨ Esto es lo que encontré")

    st.markdown(
        f'<div class="result-box">'
        f'{st.session_state.analysis_result}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.download_button(
        label="⬇️ Descargar análisis (.txt)",
        data=st.session_state.analysis_result,
        file_name="analisis_imagen.txt",
        mime="text/plain",
    )


# ==================================================
# PIE DE PÁGINA
# ==================================================

st.markdown("""
<div class="footer">
    VisionAI · Powered by Streamlit & OpenAI
    <br>Inteligencia artificial aplicada al análisis visual.
</div>
""", unsafe_allow_html=True)
