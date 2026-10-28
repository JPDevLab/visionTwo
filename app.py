
import io
import base64

import streamlit as st
from openai import OpenAI
from PIL import Image, ImageOps


# --------------------------------------------------
# Configuración
# --------------------------------------------------

st.set_page_config(
    page_title="VisionAI | Análisis de imágenes",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos discretos
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.035em;
        }

        .subtitle {
            color: #64748b;
            font-size: 1rem;
            margin-top: -0.5rem;
            margin-bottom: 2rem;
        }

        .section-label {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        div[data-testid="stFileUploader"] {
            border-radius: 8px;
        }

        div.stButton > button {
            width: 100%;
            min-height: 2.8rem;
            font-weight: 500;
            border-radius: 6px;
        }

        div[data-testid="stSidebar"] {
            border-right: 1px solid #e5e7eb;
        }

        .footer-note {
            color: #94a3b8;
            font-size: 0.8rem;
            margin-top: 2rem;
        }

        hr {
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
            border-color: #e5e7eb;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Funciones
# --------------------------------------------------

def prepare_image(uploaded_file):
    """Convierte la imagen a JPEG y la codifica en Base64."""
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)

    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)

    encoded_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return image, encoded_image


def analyze_image(api_key, encoded_image, analysis_type, context):
    """Envía la imagen a GPT-4o y devuelve el análisis."""

    client = OpenAI(api_key=api_key)

    prompts = {
        "Descripción general": (
            "Describe en español el contenido de la imagen. "
            "Identifica los elementos principales, el entorno "
            "y los detalles visuales relevantes."
        ),
        "Análisis detallado": (
            "Realiza un análisis detallado en español. "
            "Describe los objetos, personas, colores, composición, "
            "relaciones entre elementos y detalles relevantes. "
            "No inventes información que no pueda observarse."
        ),
        "Lectura de texto": (
            "Identifica y transcribe en español el texto visible "
            "en la imagen. Conserva, en lo posible, su estructura. "
            "Si hay texto ilegible, indícalo."
        ),
        "Pregunta personalizada": (
            "Responde en español a la pregunta o instrucción "
            "del usuario sobre la imagen. Distingue claramente "
            "lo observable de cualquier inferencia."
        ),
    }

    prompt = prompts[analysis_type]

    if context.strip():
        prompt += (
            "\n\nInstrucciones adicionales del usuario:\n"
            + context.strip()
        )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                + encoded_image
                            )
                        },
                    },
                ],
            }
        ],
        max_tokens=800,
    )

    return response.choices[0].message.content or (
        "No se recibió una respuesta."
    )


# --------------------------------------------------
# Barra lateral
# --------------------------------------------------

with st.sidebar:
    st.title("VisionAI")
    st.caption("Herramienta de análisis visual")
    st.divider()

    st.subheader("Configuración")

    api_key = st.text_input(
        "Clave API de OpenAI",
        type="password",
        help="La clave se utiliza para realizar el análisis.",
        placeholder="sk-...",
    )

    st.caption(
        "Necesitas una clave API válida de OpenAI "
        "para utilizar el servicio."
    )

    st.divider()

    st.caption("Modelo utilizado")
    st.code("gpt-4o", language=None)

    st.caption("Los análisis pueden tener un costo según "
               "el uso de la API.")


# --------------------------------------------------
# Encabezado
# --------------------------------------------------

st.title("Análisis de imágenes")

st.markdown(
    """
    <p class="subtitle">
        Carga una imagen, selecciona el tipo de análisis
        y consulta los resultados generados por inteligencia artificial.
    </p>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Entrada y configuración del análisis
# --------------------------------------------------

image_column, settings_column = st.columns(
    [1.15, 0.85],
    gap="large",
)

with image_column:
    st.markdown(
        '<p class="section-label">01 — Imagen de entrada</p>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png", "webp"],
        help="Formatos admitidos: JPG, PNG y WEBP.",
    )

    if uploaded_file:
        try:
            preview_image = Image.open(uploaded_file)
            preview_image = ImageOps.exif_transpose(preview_image)

            st.image(
                preview_image,
                use_container_width=True,
            )

            st.caption(
                f"Archivo: {uploaded_file.name} · "
                f"{preview_image.width} × {preview_image.height} px"
            )

        except Exception:
            st.error("No fue posible abrir el archivo seleccionado.")

    else:
        st.info(
            "Selecciona una imagen para visualizarla "
            "y preparar el análisis."
        )


with settings_column:
    st.markdown(
        '<p class="section-label">02 — Parámetros del análisis</p>',
        unsafe_allow_html=True,
    )

    analysis_type = st.selectbox(
        "Tipo de análisis",
        [
            "Descripción general",
            "Análisis detallado",
            "Lectura de texto",
            "Pregunta personalizada",
        ],
    )

    if analysis_type == "Pregunta personalizada":
        context_label = "Pregunta sobre la imagen"
        context_placeholder = (
            "Ejemplo: ¿Qué objetos aparecen en primer plano?"
        )
    else:
        context_label = "Contexto adicional (opcional)"
        context_placeholder = (
            "Añade información o instrucciones para orientar "
            "el análisis."
        )

    context = st.text_area(
        context_label,
        placeholder=context_placeholder,
        height=150,
    )

    st.caption(
        "El contexto ayuda a orientar la respuesta, "
        "pero no garantiza que todos los detalles sean correctos."
    )

    analyze_button = st.button(
        "Analizar imagen",
        type="primary",
        disabled=not bool(uploaded_file and api_key.strip()),
    )


# --------------------------------------------------
# Procesamiento
# --------------------------------------------------

if analyze_button:
    if analysis_type == "Pregunta personalizada" and not context.strip():
        st.warning("Escribe la pregunta que quieres realizar.")
    else:
        try:
            with st.spinner("Analizando la imagen..."):
                _, encoded_image = prepare_image(uploaded_file)

                result = analyze_image(
                    api_key=api_key.strip(),
                    encoded_image=encoded_image,
                    analysis_type=analysis_type,
                    context=context,
                )

                st.session_state["analysis_result"] = result
                st.session_state["analysis_filename"] = uploaded_file.name
                st.session_state["analysis_type"] = analysis_type

        except Exception as error:
            st.error(
                "No fue posible completar el análisis. "
                "Verifica la clave API, tu conexión y los detalles "
                "del error."
            )
            st.caption(str(error))


# --------------------------------------------------
# Resultados
# --------------------------------------------------

if st.session_state.get("analysis_result"):
    st.divider()

    st.markdown(
        '<p class="section-label">03 — Resultado</p>',
        unsafe_allow_html=True,
    )

    st.subheader(
        st.session_state.get(
            "analysis_filename",
            "Resultado del análisis",
        )
    )

    st.caption(
        "Tipo de análisis: "
        + st.session_state.get("analysis_type", "No especificado")
    )

    st.markdown(st.session_state["analysis_result"])

    st.download_button(
        label="Descargar resultado (.txt)",
        data=st.session_state["analysis_result"],
        file_name="resultado_visionai.txt",
        mime="text/plain",
    )

st.markdown(
    """
    <p class="footer-note">
        VisionAI · Análisis de imágenes mediante OpenAI
    </p>
    """,
    unsafe_allow_html=True,
)
