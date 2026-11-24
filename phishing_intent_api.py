"""
API de Análisis de Intenciones y Detección de Phishing mediante PLN Básico.

Recibe archivos .txt vía HTTP (FastAPI), aplica un pipeline de Procesamiento de
Lenguaje Natural (normalización, tokenización, eliminación de stopwords, extracción
de n-gramas, análisis léxico y heurísticas de URLs) y devuelve:
  - Intenciones detectadas con puntuación y evidencia.
  - Alertas de seguridad clasificadas por severidad.
  - Probabilidad de phishing (0.0 a 1.0) y nivel de riesgo.

Ejecución local:
    pip install fastapi uvicorn python-multipart
    uvicorn phishing_intent_api:app --reload

Documentación interactiva (Swagger UI):
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field


# =============================================================================
# 1. Recursos lingüísticos y reglas de PLN básico (español + inglés común)
# =============================================================================

STOPWORDS_ES = {
    "a", "acá", "ahí", "al", "algo", "algunas", "algunos", "allá", "allí",
    "ante", "antes", "aquel", "aquella", "aquellas", "aquellos", "aquí", "así",
    "aun", "aunque", "bajo", "bien", "cada", "casi", "como", "con", "contra",
    "cual", "cuales", "cuando", "cuanto", "de", "del", "desde", "donde", "dos",
    "el", "ella", "ellas", "ellos", "en", "entre", "era", "eran", "es", "esa",
    "esas", "ese", "eso", "esos", "esta", "estaba", "estado", "estan", "estar",
    "estas", "este", "esto", "estos", "fue", "fueron", "ha", "haber", "había",
    "hace", "hacen", "hacer", "hacia", "han", "hasta", "hay", "la", "las", "le",
    "les", "lo", "los", "mas", "me", "mi", "mis", "mucho", "muy", "nada", "ni",
    "no", "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro",
    "nuestros", "o", "os", "otra", "otras", "otro", "otros", "para", "pero",
    "poco", "por", "porque", "que", "quien", "quienes", "se", "sea", "segun",
    "ser", "si", "sido", "siempre", "sin", "sobre", "sois", "somos", "son",
    "soy", "su", "sus", "tambien", "tan", "te", "tengo", "ti", "tiene",
    "tienen", "toda", "todas", "todo", "todos", "tu", "tus", "un", "una",
    "unas", "uno", "unos", "usted", "ustedes", "va", "vamos", "van", "ya", "y",
    # Stopwords frecuentes en inglés por si el texto es mixto
    "the", "and", "for", "that", "this", "with", "you", "your", "from", "are",
    "was", "were", "have", "has", "had", "not", "but", "can", "will", "just",
}

INTENT_PATTERNS: Dict[str, Dict[str, object]] = {
    "solicitud_credenciales": {
        "descripcion": "Solicita contraseñas, códigos de verificación, PIN o acceso a cuentas.",
        "peso_phishing": 0.34,
        "patrones": [
            r"\b(contrasena|password|clave|pin|nip|token|codigo\s+de\s+verificacion|otp)\b",
            r"\b(inicia\s+sesion|iniciar\s+sesion|verifica\s+tu\s+cuenta|validar\s+identidad|valida\s+su\s+identidad)\b",
            r"\b(confirma\s+tus\s+datos|actualiza\s+tus\s+datos|restablece\s+tu\s+clave)\b",
            r"\b(ingresa\s+tus\s+credenciales|introduce\s+tu\s+usuario|verify\s+your\s+account)\b",
        ],
    },
    "solicitud_pago_o_dinero": {
        "descripcion": "Pide transferencias, datos bancarios, tarjetas o pagos urgentes.",
        "peso_phishing": 0.30,
        "patrones": [
            r"\b(tarjeta\s+de\s+credito|tarjeta\s+de\s+debito|cvv|clabe|numero\s+de\s+cuenta)\b",
            r"\b(transferencia|deposito|pago\s+pendiente|factura\s+vencida|adeudo|multa)\b",
            r"\b(datos\s+bancarios|informacion\s+de\s+pago|actualizar\s+metodo\s+de\s+pago)\b",
            r"\b(criptomoneda|bitcoin|usdt|reembolso\s+pendiente|liberar\s+paquete)\b",
        ],
    },
    "urgencia_o_presion": {
        "descripcion": "Emplea presión temporal o amenazas de suspensión para forzar acción rápida.",
        "peso_phishing": 0.24,
        "patrones": [
            r"\b(urgente|inmediato|inmediatamente|ahora\s+mismo|de\s+inmediato|accion\s+requerida)\b",
            r"\b(ultima\s+advertencia|ultimo\s+aviso|expira\s+hoy|vence\s+hoy|en\s+las\s+proximas\s+\d+\s+horas)\b",
            r"\b(cuenta\s+suspendida|sera\s+bloqueada|sera\s+cancelada|cierre\s+definitivo|bloqueo\s+inminente)\b",
            r"\b(evita\s+la\s+suspension|evitar\s+cargos|actividad\s+sospechosa|acceso\s+no\s+autorizado)\b",
        ],
    },
    "redireccion_enlace": {
        "descripcion": "Induce a hacer clic en enlaces externos o descargar archivos.",
        "peso_phishing": 0.20,
        "patrones": [
            r"\b(haz\s+clic\s+aqui|da\s+clic|presiona\s+aqui|ingresa\s+al\s+enlace|abre\s+el\s+enlace)\b",
            r"\b(accede\s+desde\s+aqui|descarga\s+el\s+archivo|adjunto|portal\s+de\s+verificacion)\b",
            r"\b(click\s+here|follow\s+this\s+link|open\s+the\s+link)\b",
            r"https?://\S+",
        ],
    },
    "suplantacion_identidad": {
        "descripcion": "Menciona entidades bancarias, gubernamentales o tecnológicas frecuentemente suplantadas.",
        "peso_phishing": 0.22,
        "patrones": [
            r"\b(sat|hacienda|imss|infonavit|correos\s+de\s+mexico|dhl|estafeta|fedex|ups)\b",
            r"\b(bbva|banorte|santander|citibanamex|banamex|hsbc|scotiabank|nu\s+mexico|paypal)\b",
            r"\b(microsoft|google|apple|amazon|mercado\s+libre|netflix|whatsapp|soporte\s+tecnico)\b",
            r"\b(departamento\s+de\s+seguridad|equipo\s+de\s+seguridad|administrador\s+del\s+sistema)\b",
        ],
    },
    "oferta_o_premio": {
        "descripcion": "Promete premios, sorteos, ganancias fáciles o reembolsos inesperados.",
        "peso_phishing": 0.22,
        "patrones": [
            r"\b(felicidades|has\s+ganado|ganador|premio|sorteo|loteria|recompensa)\b",
            r"\b(gratis|100%\s+gratuito|oferta\s+exclusiva|regalo|bono\s+especial|reclama\s+tu)\b",
            r"\b(herencia|dinero\s+facil|ganancias\s+garantizadas)\b",
        ],
    },
    "consulta_o_soporte": {
        "descripcion": "Plantea una duda, solicitud de información o seguimiento regular.",
        "peso_phishing": -0.10,
        "patrones": [
            r"\b(podrias\s+ayudarme|tengo\s+una\s+duda|quisiera\s+saber|me\s+puedes\s+indicar)\b",
            r"\b(consulta|pregunta|soporte|cotizacion|horario|informacion\s+sobre)\b",
            r"\?",
        ],
    },
    "informativo_o_notificacion": {
        "descripcion": "Comunica un aviso general, agenda, minuta o actualización rutinaria.",
        "peso_phishing": -0.08,
        "patrones": [
            r"\b(reunion|junta|calendario|minuta|resumen|reporte\s+semanal|actualizacion)\b",
            r"\b(les\s+comparto|te\s+comparto|adjunto\s+el\s+documento|recordatorio\s+de\s+la\s+sesion)\b",
            r"\b(saludos\s+cordiales|quedo\s+atento|buen\s+dia|buenas\s+tardes)\b",
        ],
    },
}

SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".tk", ".ml", ".ga", ".cf", ".gq", ".click", ".link",
    ".work", ".support", ".info", ".ru", ".cn", ".pw", ".buzz", ".loan",
    ".cam", ".rest", ".icu",
}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc", "rebrand.ly",
}

IMPERSONATED_BRANDS = {
    "bbva", "banorte", "santander", "banamex", "citibanamex", "hsbc",
    "scotiabank", "paypal", "sat", "imss", "amazon", "mercadolibre",
    "microsoft", "google", "apple", "netflix", "dhl", "fedex", "estafeta",
}

URL_REGEX = re.compile(r"(https?://[^\s<>\"'\)\]]+)", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,3}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}\b")
IP_HOST_REGEX = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


# =============================================================================
# 2. Modelos Pydantic de respuesta
# =============================================================================

class MetadatosArchivo(BaseModel):
    nombre: str
    codificacion_detectada: str
    caracteres: int
    palabras: int
    oraciones: int


class ResumenNLP(BaseModel):
    palabras_clave: List[str] = Field(description="Términos más frecuentes tras eliminar stopwords.")
    bigramas_frecuentes: List[str] = Field(description="Pares de palabras consecutivas relevantes.")
    urls_detectadas: List[str]
    correos_detectados: List[str]
    telefonos_detectados: List[str]
    proporcion_mayusculas: float = Field(description="Fracción de letras en mayúsculas (0 a 1).")
    signos_exclamacion: int


class IntencionDetectada(BaseModel):
    nombre: str
    probabilidad: float = Field(ge=0.0, le=1.0)
    descripcion: str
    evidencia: List[str]


class AlertaSeguridad(BaseModel):
    codigo: str
    severidad: Literal["baja", "media", "alta"]
    mensaje: str
    evidencia: List[str]


class ResultadoAnalisis(BaseModel):
    archivo: Optional[MetadatosArchivo] = None
    intencion_principal: str
    intenciones: List[IntencionDetectada]
    alertas: List[AlertaSeguridad]
    probabilidad_phishing: float = Field(ge=0.0, le=1.0)
    porcentaje_phishing: float = Field(ge=0.0, le=100.0)
    nivel_riesgo: Literal["bajo", "medio", "alto", "critico"]
    procesamiento_nlp: ResumenNLP
    resumen: str


class SolicitudTextoPlano(BaseModel):
    texto: str = Field(..., min_length=1, description="Texto a analizar directamente.")


# =============================================================================
# 3. Pipeline de Procesamiento de Lenguaje Natural (PLN básico)
# =============================================================================

def normalizar_texto(texto: str) -> str:
    """Convierte a minúsculas y elimina acentos/diacríticos para búsqueda léxica."""
    nfkd = unicodedata.normalize("NFKD", texto.lower())
    sin_acentos = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", sin_acentos).strip()


def tokenizar_palabras(texto_normalizado: str) -> List[str]:
    """Extrae tokens alfabéticos eliminando URLs y signos de puntuación."""
    sin_urls = URL_REGEX.sub(" ", texto_normalizado)
    return re.findall(r"[a-zñ]{2,}", sin_urls)


def dividir_oraciones(texto: str) -> List[str]:
    """Divide el texto en oraciones básicas por signos de puntuación o saltos de línea."""
    partes = re.split(r"[.!?\n]+", texto)
    return [p.strip() for p in partes if p.strip()]


def extraer_palabras_clave_y_bigramas(tokens: List[str], top_n: int = 8) -> Tuple[List[str], List[str]]:
    """Filtra stopwords y calcula los unigramas y bigramas más frecuentes."""
    limpios = [t for t in tokens if t not in STOPWORDS_ES and len(t) > 2]
    unigramas = [palabra for palabra, _ in Counter(limpios).most_common(top_n)]

    bigramas_raw = [f"{limpios[i]} {limpios[i + 1]}" for i in range(len(limpios) - 1)]
    bigramas = [bg for bg, freq in Counter(bigramas_raw).most_common(5) if freq >= 1]
    return unigramas, bigramas


def analizar_urls(urls: List[str], texto_normalizado: str) -> Tuple[List[AlertaSeguridad], float]:
    """Evalúa características sospechosas en URLs y devuelve alertas y puntaje adicional."""
    alertas: List[AlertaSeguridad] = []
    puntaje_url = 0.0

    urls_http = []
    urls_ip = []
    urls_acortadas = []
    urls_tld_sospechoso = []
    urls_con_arroba_o_punycode = []
    urls_suplantacion_marca = []

    for raw_url in urls:
        parsed = urlparse(raw_url)
        host = (parsed.hostname or "").lower()

        if parsed.scheme == "http":
            urls_http.append(raw_url)

        if IP_HOST_REGEX.match(host):
            urls_ip.append(raw_url)

        if host in URL_SHORTENERS or any(host.endswith("." + s) for s in URL_SHORTENERS):
            urls_acortadas.append(raw_url)

        if any(host.endswith(tld) for tld in SUSPICIOUS_TLDS):
            urls_tld_sospechoso.append(raw_url)

        if "@" in raw_url or "xn--" in host or host.count(".") >= 3:
            urls_con_arroba_o_punycode.append(raw_url)

        # Marca mencionada en el texto o en subdominio, pero el dominio raíz no coincide
        partes_host = host.split(".")
        dominio_raiz = ".".join(partes_host[-2:]) if len(partes_host) >= 2 else host
        for marca in IMPERSONATED_BRANDS:
            if marca in texto_normalizado and marca in raw_url.lower() and not dominio_raiz.startswith(marca + "."):
                urls_suplantacion_marca.append(raw_url)
                break

    if urls_ip:
        puntaje_url += 0.35
        alertas.append(AlertaSeguridad(
            codigo="URL_DIRECCION_IP",
            severidad="alta",
            mensaje="Se detectó un enlace apuntando directamente a una dirección IP en lugar de un dominio legítimo.",
            evidencia=urls_ip,
        ))

    if urls_suplantacion_marca:
        puntaje_url += 0.30
        alertas.append(AlertaSeguridad(
            codigo="DOMINIO_SUPLANTADOR",
            severidad="alta",
            mensaje="Un enlace usa el nombre de una marca conocida fuera de su dominio principal.",
            evidencia=urls_suplantacion_marca,
        ))

    if urls_acortadas:
        puntaje_url += 0.18
        alertas.append(AlertaSeguridad(
            codigo="URL_ACORTADA",
            severidad="media",
            mensaje="El texto contiene enlaces acortados que ocultan el destino real.",
            evidencia=urls_acortadas,
        ))

    if urls_tld_sospechoso:
        puntaje_url += 0.22
        alertas.append(AlertaSeguridad(
            codigo="TLD_SOSPECHOSO",
            severidad="alta",
            mensaje="Se detectaron dominios con extensiones (TLD) frecuentemente asociadas a phishing.",
            evidencia=urls_tld_sospechoso,
        ))

    if urls_con_arroba_o_punycode:
        puntaje_url += 0.25
        alertas.append(AlertaSeguridad(
            codigo="URL_OFUSCADA",
            severidad="alta",
            mensaje="Un enlace contiene símbolos '@', punycode (xn--) o subdominios excesivos para confundir al usuario.",
            evidencia=urls_con_arroba_o_punycode,
        ))

    if urls_http:
        puntaje_url += 0.10
        alertas.append(AlertaSeguridad(
            codigo="ENLACE_NO_CIFRADO",
            severidad="baja",
            mensaje="Se encontraron enlaces HTTP sin cifrado TLS/HTTPS.",
            evidencia=urls_http,
        ))

    return alertas, min(puntaje_url, 0.55)


def clasificar_intenciones(texto_normalizado: str) -> Tuple[List[IntencionDetectada], Dict[str, float]]:
    """Detecta intenciones mediante coincidencia de patrones léxicos y calcula probabilidades."""
    puntajes_brutos: Dict[str, float] = {}
    evidencias_por_intencion: Dict[str, List[str]] = {}

    for nombre, config in INTENT_PATTERNS.items():
        coincidencias: List[str] = []
        patrones: List[str] = config["patrones"]  # type: ignore[assignment]
        for patron in patrones:
            for match in re.finditer(patron, texto_normalizado, flags=re.IGNORECASE):
                fragmento = match.group(0).strip()
                if fragmento and fragmento not in coincidencias:
                    coincidencias.append(fragmento)

        if coincidencias:
            # Escala saturada suave: 1 match -> ~0.44, 2 -> ~0.69, 3 -> ~0.82, 4+ -> hasta 0.95
            score = round(min(0.95, 1.0 - math.exp(-0.58 * len(coincidencias))), 4)
            puntajes_brutos[nombre] = score
            evidencias_por_intencion[nombre] = coincidencias[:6]

    if not puntajes_brutos:
        intencion_default = IntencionDetectada(
            nombre="informativo_o_notificacion",
            probabilidad=0.50,
            descripcion="Texto general sin señales claras de riesgo o acción específica.",
            evidencia=[],
        )
        return [intencion_default], {"informativo_o_notificacion": 0.50}

    intenciones = [
        IntencionDetectada(
            nombre=nombre,
            probabilidad=score,
            descripcion=str(INTENT_PATTERNS[nombre]["descripcion"]),
            evidencia=evidencias_por_intencion[nombre],
        )
        for nombre, score in sorted(puntajes_brutos.items(), key=lambda x: x[1], reverse=True)
    ]
    return intenciones, puntajes_brutos


def analizar_texto(texto: str, metadatos: Optional[MetadatosArchivo] = None) -> ResultadoAnalisis:
    """Ejecuta el flujo completo de PLN, detección de intenciones, alertas y phishing."""
    texto_normalizado = normalizar_texto(texto)
    tokens = tokenizar_palabras(texto_normalizado)
    oraciones = dividir_oraciones(texto)
    palabras_clave, bigramas = extraer_palabras_clave_y_bigramas(tokens)

    urls = list(dict.fromkeys(URL_REGEX.findall(texto)))
    correos = list(dict.fromkeys(EMAIL_REGEX.findall(texto)))
    telefonos = list(dict.fromkeys(PHONE_REGEX.findall(texto)))

    letras = [c for c in texto if c.isalpha()]
    prop_mayusculas = round(sum(1 for c in letras if c.isupper()) / len(letras), 4) if letras else 0.0
    signos_exclamacion = texto.count("!")

    intenciones, puntajes_intenciones = clasificar_intenciones(texto_normalizado)
    intencion_principal = intenciones[0].nombre

    # --- Construcción de alertas basadas en intenciones y estilo ---
    alertas: List[AlertaSeguridad] = []
    evidencia_por_nombre = {i.nombre: i.evidencia for i in intenciones}

    if "solicitud_credenciales" in puntajes_intenciones:
        alertas.append(AlertaSeguridad(
            codigo="SOLICITUD_CREDENCIALES",
            severidad="alta",
            mensaje="El texto solicita contraseñas, códigos de verificación o validación de acceso.",
            evidencia=evidencia_por_nombre["solicitud_credenciales"],
        ))

    if "solicitud_pago_o_dinero" in puntajes_intenciones:
        alertas.append(AlertaSeguridad(
            codigo="SOLICITUD_FINANCIERA",
            severidad="alta",
            mensaje="Se solicitan datos bancarios, tarjetas o pagos que podrían ser fraudulentos.",
            evidencia=evidencia_por_nombre["solicitud_pago_o_dinero"],
        ))

    if "urgencia_o_presion" in puntajes_intenciones:
        alertas.append(AlertaSeguridad(
            codigo="URGENCIA_ARTIFICIAL",
            severidad="media" if puntajes_intenciones["urgencia_o_presion"] < 0.65 else "alta",
            mensaje="El mensaje utiliza lenguaje de urgencia o amenaza de bloqueo para presionar una respuesta rápida.",
            evidencia=evidencia_por_nombre["urgencia_o_presion"],
        ))

    if "suplantacion_identidad" in puntajes_intenciones and (
        "solicitud_credenciales" in puntajes_intenciones
        or "solicitud_pago_o_dinero" in puntajes_intenciones
        or "redireccion_enlace" in puntajes_intenciones
    ):
        alertas.append(AlertaSeguridad(
            codigo="POSIBLE_SUPLANTACION",
            severidad="alta",
            mensaje="Menciona una marca o institución reconocida junto con solicitudes sensibles o enlaces.",
            evidencia=evidencia_por_nombre["suplantacion_identidad"],
        ))

    if "oferta_o_premio" in puntajes_intenciones:
        alertas.append(AlertaSeguridad(
            codigo="PROMESA_PREMIO_O_RECOMPENSA",
            severidad="media",
            mensaje="Ofrece premios, regalos o recompensas inesperadas, patrón común en ingeniería social.",
            evidencia=evidencia_por_nombre["oferta_o_premio"],
        ))

    if prop_mayusculas >= 0.35 and len(letras) >= 20:
        alertas.append(AlertaSeguridad(
            codigo="EXCESO_MAYUSCULAS",
            severidad="baja",
            mensaje="Uso elevado de mayúsculas sostenidas, típico de mensajes alarmistas.",
            evidencia=[f"proporcion_mayusculas={prop_mayusculas:.0%}"],
        ))

    if signos_exclamacion >= 3:
        alertas.append(AlertaSeguridad(
            codigo="EXCESO_EXCLAMACIONES",
            severidad="baja",
            mensaje="Uso reiterado de signos de exclamación para generar alarma o entusiasmo artificial.",
            evidencia=[f"signos_exclamacion={signos_exclamacion}"],
        ))

    # --- Análisis heurístico de URLs ---
    alertas_url, puntaje_url = analizar_urls(urls, texto_normalizado)
    alertas.extend(alertas_url)

    # --- Cálculo de probabilidad de phishing (modelo logístico ponderado) ---
    # Sesgo base negativo: un texto neutro sin señales de riesgo tiene baja probabilidad (~4%).
    logit = -3.10

    for nombre, score in puntajes_intenciones.items():
        peso = float(INTENT_PATTERNS[nombre]["peso_phishing"])
        logit += peso * score * 4.5

    # Contribución de heurísticas de URL
    logit += puntaje_url * 4.2

    # Señales estilísticas
    if prop_mayusculas >= 0.35 and len(letras) >= 20:
        logit += 0.45
    if signos_exclamacion >= 3:
        logit += 0.35

    # Sinergia de phishing clásico: (urgencia o suplantación) + (credenciales/pago/enlace)
    tiene_presion = (
        "urgencia_o_presion" in puntajes_intenciones
        or "suplantacion_identidad" in puntajes_intenciones
    )
    tiene_accion_riesgosa = (
        "solicitud_credenciales" in puntajes_intenciones
        or "solicitud_pago_o_dinero" in puntajes_intenciones
        or bool(urls)
    )
    if tiene_presion and tiene_accion_riesgosa:
        logit += 1.15

    probabilidad_phishing = round(1.0 / (1.0 + math.exp(-logit)), 4)
    porcentaje_phishing = round(probabilidad_phishing * 100.0, 2)

    if probabilidad_phishing >= 0.85:
        nivel_riesgo: Literal["bajo", "medio", "alto", "critico"] = "critico"
    elif probabilidad_phishing >= 0.60:
        nivel_riesgo = "alto"
    elif probabilidad_phishing >= 0.30:
        nivel_riesgo = "medio"
    else:
        nivel_riesgo = "bajo"

    # Ordenar alertas por severidad (alta -> media -> baja)
    prioridad = {"alta": 0, "media": 1, "baja": 2}
    alertas.sort(key=lambda a: prioridad[a.severidad])

    resumen = (
        f"Intención principal detectada: '{intencion_principal}'. "
        f"Probabilidad estimada de phishing: {porcentaje_phishing:.2f}% (riesgo {nivel_riesgo}). "
        f"Se generaron {len(alertas)} alerta(s) de seguridad."
    )

    if metadatos is None:
        metadatos = MetadatosArchivo(
            nombre="texto_directo.txt",
            codificacion_detectada="utf-8",
            caracteres=len(texto),
            palabras=len(tokens),
            oraciones=len(oraciones),
        )

    return ResultadoAnalisis(
        archivo=metadatos,
        intencion_principal=intencion_principal,
        intenciones=intenciones,
        alertas=alertas,
        probabilidad_phishing=probabilidad_phishing,
        porcentaje_phishing=porcentaje_phishing,
        nivel_riesgo=nivel_riesgo,
        procesamiento_nlp=ResumenNLP(
            palabras_clave=palabras_clave,
            bigramas_frecuentes=bigramas,
            urls_detectadas=urls,
            correos_detectados=correos,
            telefonos_detectados=telefonos,
            proporcion_mayusculas=prop_mayusculas,
            signos_exclamacion=signos_exclamacion,
        ),
        resumen=resumen,
    )


def decodificar_bytes_txt(contenido: bytes) -> Tuple[str, str]:
    """Intenta decodificar el archivo .txt probando UTF-8 (con/sin BOM) y Latin-1."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return contenido.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise HTTPException(status_code=400, detail="No se pudo decodificar el archivo .txt como texto válido.")


# =============================================================================
# 4. Aplicación FastAPI y endpoints
# =============================================================================

app = FastAPI(
    title="API de Análisis de Intenciones y Detección de Phishing",
    description=(
        "Analiza archivos .txt mediante Procesamiento de Lenguaje Natural básico "
        "para identificar intenciones, alertas de seguridad y probabilidad de phishing."
    ),
    version="1.0.0",
)


@app.get("/health", tags=["Estado"])
def health_check() -> Dict[str, str]:
    """Verifica que la API esté activa."""
    return {"status": "ok"}


@app.post("/analizar", response_model=ResultadoAnalisis, tags=["Análisis"])
async def analizar_archivo_txt(archivo: UploadFile = File(..., description="Archivo .txt a analizar")) -> ResultadoAnalisis:
    """Recibe un archivo .txt, ejecuta el análisis de PLN y devuelve intenciones, alertas y riesgo de phishing."""
    nombre_archivo = archivo.filename or ""
    if not nombre_archivo.lower().endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Solo se permiten archivos con extensión .txt.",
        )

    contenido_bytes = await archivo.read()
    if not contenido_bytes.strip():
        raise HTTPException(
            status_code=400,
            detail="El archivo .txt está vacío.",
        )

    # Límite de seguridad de 2 MB para archivos de texto plano
    if len(contenido_bytes) > 2 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="El archivo excede el tamaño máximo permitido de 2 MB.",
        )

    texto, encoding = decodificar_bytes_txt(contenido_bytes)
    texto_norm = normalizar_texto(texto)
    metadatos = MetadatosArchivo(
        nombre=nombre_archivo,
        codificacion_detectada=encoding,
        caracteres=len(texto),
        palabras=len(tokenizar_palabras(texto_norm)),
        oraciones=len(dividir_oraciones(texto)),
    )
    return analizar_texto(texto, metadatos=metadatos)


@app.post("/analizar-texto", response_model=ResultadoAnalisis, tags=["Análisis"])
def analizar_texto_plano(payload: SolicitudTextoPlano) -> ResultadoAnalisis:
    """Endpoint auxiliar para probar texto directamente en JSON sin subir un archivo."""
    if not payload.texto.strip():
        raise HTTPException(status_code=400, detail="El campo 'texto' no puede estar vacío.")
    return analizar_texto(payload.texto)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("phishing_intent_api:app", host="127.0.0.1", port=8000, reload=True)