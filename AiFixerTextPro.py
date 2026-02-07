#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Text Fixer PRO (ES) — Perfiles Docente / LinkedIn / Helpdesk + IA Check Control

✔ Limpieza avanzada: espacios, puntuación, comillas, elipsis, unidades, porcentajes.
✔ Reducción "robotese IA": muletillas, frases hinchadas, redundancias.
✔ Detección y eliminación de repeticiones (líneas y frases).
✔ Split inteligente de frases kilométricas.
✔ Perfiles de estilo:
   - neutro: normaliza sin cambiar intención
   - docente: claridad pedagógica + estructura + ejemplos
   - linkedin: ritmo + gancho + CTA suave + párrafos cortos + emojis moderados
   - helpdesk: formato ticket, pasos numerados, tono profesional y directo
✔ IA Check Control:
   - AI-likeness score (0-100)
   - alertas de patrones típicos
   - métricas: longitud frases, repetición, conectores, boilerplate
✔ Opcional: corrección gramatical con LanguageTool (si está instalado)

Uso:
  python ai_text_fixer_pro.py --in texto.txt --out texto_ok.txt --profile linkedin --report --ai-check
  cat texto.txt | python ai_text_fixer_pro.py --stdin --profile helpdesk --report --ai-check > salida.txt
  python ai_text_fixer_pro.py --in texto.txt --out texto_ok.txt --profile docente --languagetool es --report

Instalar LanguageTool (opcional):
  pip install language-tool-python
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Patrones típicos "IA-robot"
# -----------------------------
AI_TICS_REGEX = [
    r"\ben conclusión\b",
    r"\ben resumen\b",
    r"\bcomo (una )?IA\b",
    r"\bcomo modelo de lenguaje\b",
    r"\bes importante destacar que\b",
    r"\bcabe destacar que\b",
    r"\ba tener en cuenta\b",
    r"\ben términos generales\b",
    r"\bde alguna manera\b",
    r"\bpor otro lado\b",
    r"\ben este sentido\b",
    r"\b(en|de) (última|último) instancia\b",
    r"\bno obstante\b",
    r"\bsin embargo\b",
    r"\bademás\b",
]

BOILERPLATE_REGEX = [
    r"\bno puedo\b.*\bpero\b",            # típico "no puedo, pero..."
    r"\bcomo IA\b",                        # autodeclaración
    r"\bno tengo acceso\b",                # autodeclaración
    r"\bno estoy seguro\b",                # autojustificación (a veces útil, pero en contenido final molesta)
]

# Simplificaciones suaves (ES)
STYLE_REWRITES = [
    (re.compile(r"\bEs importante destacar que\s+", re.IGNORECASE), ""),
    (re.compile(r"\bCabe destacar que\s+", re.IGNORECASE), ""),
    (re.compile(r"\bEn términos generales,\s*", re.IGNORECASE), ""),
    (re.compile(r"\bDe alguna manera,\s*", re.IGNORECASE), ""),
    (re.compile(r"\bEn este sentido,\s*", re.IGNORECASE), ""),
]

# Reducción de conectores repetidos
DOUBLE_CONNECTORS = [
    (re.compile(r"\bAdemás,\s*además\b", re.IGNORECASE), "Además"),
    (re.compile(r"\bSin embargo,\s*sin embargo\b", re.IGNORECASE), "Sin embargo"),
    (re.compile(r"\bNo obstante,\s*no obstante\b", re.IGNORECASE), "No obstante"),
]


# -----------------------------
# Normalizaciones técnicas
# -----------------------------
RE_SPACES = re.compile(r"[ \t]{2,}")
RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
RE_MISSING_SPACE_AFTER_PUNCT = re.compile(r"([,.;:!?])([^\s\)\]\}»”])")
RE_ELLIPSIS = re.compile(r"\.{4,}")
RE_DOUBLE_PUNCT = re.compile(r"([!?]){2,}")
RE_DASHES = re.compile(r"\s*-\s*")  # rango simple 1-2 (no guion largo)
RE_PERCENT = re.compile(r"(\d)\s*%")
RE_UNITS = re.compile(r"(\d)\s*(km|cm|mm|m|kg|g|mb|gb|tb)\b", re.IGNORECASE)

QUOTE_MAP = {
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
}


# -----------------------------
# Reporte y métricas
# -----------------------------
@dataclass
class FixReport:
    original_len: int = 0
    final_len: int = 0
    changes: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    ai_flags: List[str] = field(default_factory=list)
    ai_score: Optional[int] = None

    def add(self, msg: str) -> None:
        self.changes.append(msg)

    def add_flag(self, msg: str) -> None:
        self.ai_flags.append(msg)

    def render(self) -> str:
        lines = []
        lines.append("=== AI Text Fixer PRO — Reporte ===")
        lines.append(f"- Longitud original: {self.original_len} caracteres")
        lines.append(f"- Longitud final:    {self.final_len} caracteres")
        delta = self.final_len - self.original_len
        lines.append(f"- Delta:             {delta:+d} caracteres")

        if self.metrics:
            lines.append("\nMétricas:")
            for k, v in sorted(self.metrics.items()):
                if isinstance(v, float):
                    lines.append(f"  • {k}: {v:.2f}")
                else:
                    lines.append(f"  • {k}: {v}")

        if self.ai_score is not None:
            lines.append(f"\nIA Check Control — AI-likeness score: {self.ai_score}/100")
            if self.ai_flags:
                lines.append("Alertas:")
                for f in self.ai_flags:
                    lines.append(f"  • {f}")
            else:
                lines.append("Alertas: (ninguna)")

        if self.changes:
            lines.append("\nCambios aplicados:")
            for c in self.changes:
                lines.append(f"  • {c}")
        else:
            lines.append("\nCambios aplicados: (ninguno)")

        return "\n".join(lines)


# -----------------------------
# Utilidades texto
# -----------------------------
def normalize_quotes(text: str, report: FixReport) -> str:
    before = text
    for k, v in QUOTE_MAP.items():
        text = text.replace(k, v)
    if text != before:
        report.add("Normalizadas comillas tipográficas.")
    return text


def normalize_whitespace(text: str, report: FixReport) -> str:
    before = text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = RE_SPACES.sub(" ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if text != before:
        report.add("Normalizados espacios/saltos de línea.")
    return text


def normalize_punctuation(text: str, report: FixReport) -> str:
    before = text
    text = RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = RE_MISSING_SPACE_AFTER_PUNCT.sub(r"\1 \2", text)
    text = RE_ELLIPSIS.sub("...", text)
    text = RE_DOUBLE_PUNCT.sub(lambda m: m.group(1), text)
    text = RE_DASHES.sub("-", text)
    text = RE_PERCENT.sub(r"\1%", text)
    text = RE_UNITS.sub(lambda m: f"{m.group(1)}{m.group(2)}", text)
    if text != before:
        report.add("Normalizada puntuación (espacios, elipsis, %, unidades, guiones).")
    return text


def reduce_ai_tics(text: str, report: FixReport) -> str:
    before = text

    # Rewrites suaves
    for pattern, repl in STYLE_REWRITES:
        text = pattern.sub(repl, text)

    for pattern, repl in DOUBLE_CONNECTORS:
        text = pattern.sub(repl, text)

    # Elimina autodeclaraciones de IA en líneas completas
    text = re.sub(r"(?im)^\s*(Como (una )?IA.*?)(\n|$)", "", text)
    text = re.sub(r"(?im)^\s*(Como modelo de lenguaje.*?)(\n|$)", "", text)

    # Reduce repetición de conectores
    text = re.sub(r"\b(sin embargo|no obstante|además)\b(\s*,?\s*\b\1\b)+", r"\1", text, flags=re.IGNORECASE)

    if text != before:
        report.add("Reducido 'robotese' típico (muletillas y redundancias).")
    return text


def dedupe_repetitions(text: str, report: FixReport) -> str:
    before = text
    lines = [ln.rstrip() for ln in text.split("\n")]
    out: List[str] = []

    def norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"[^\w\sáéíóúüñ]", "", s)
        return s

    removed = 0
    prev = ""
    for ln in lines:
        n = norm(ln)
        if n and prev and n == prev:
            removed += 1
            continue
        out.append(ln)
        prev = n

    text = "\n".join(out)

    # Repetición de palabra >=3 veces seguidas: "muy muy muy" -> "muy"
    text2 = re.sub(r"\b(\w+)(\s+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)

    if text2 != text:
        removed += 1
        text = text2

    if text != before:
        report.add(f"Eliminadas repeticiones obvias: {removed} ajustes.")
    return text


def split_long_sentences(text: str, report: FixReport, max_len: int = 220) -> str:
    before = text
    paras = text.split("\n")
    new_paras: List[str] = []
    cuts = 0

    for p in paras:
        if len(p) <= max_len:
            new_paras.append(p)
            continue

        # Corte por conectores
        parts = re.split(r"(\s+(?:y|pero|aunque|sin embargo|no obstante|además|por tanto|por eso)\s+)", p, flags=re.IGNORECASE)
        chunks: List[str] = []
        buff = ""
        for part in parts:
            if len(buff) + len(part) <= max_len:
                buff += part
            else:
                if buff.strip():
                    chunks.append(buff.strip())
                    cuts += 1
                buff = part
        if buff.strip():
            chunks.append(buff.strip())

        # Si aún demasiado largo, corta por comas
        final_lines: List[str] = []
        for c in chunks:
            if len(c) <= max_len:
                final_lines.append(c)
            else:
                comma_parts = [x.strip() for x in c.split(",")]
                tmp = ""
                for cp in comma_parts:
                    piece = (cp + ", ").strip()
                    if len(tmp) + len(piece) <= max_len:
                        tmp += piece
                    else:
                        if tmp.strip():
                            final_lines.append(tmp.rstrip(", ").strip())
                            cuts += 1
                        tmp = piece
                if tmp.strip():
                    final_lines.append(tmp.rstrip(", ").strip())

        new_paras.append("\n".join(final_lines))

    text = "\n".join(new_paras)
    if text != before:
        report.add(f"Divididas frases demasiado largas: {cuts} cortes.")
    return text


# -----------------------------
# Perfiles de estilo
# -----------------------------
def apply_profile_neutro(text: str, report: FixReport) -> str:
    # Muy poco invasivo
    return text


def apply_profile_docente(text: str, report: FixReport) -> str:
    """
    Docente: claridad + microestructura sin inventar contenido.
    - Refuerza enumeraciones si detecta "Paso 1/2/3"
    - Sugiere formato de objetivos/ejemplo si ve definiciones
    """
    before = text

    # Si el texto tiene muchos puntos y seguido, fomenta párrafos
    text = re.sub(r"(?<!\n)\n?(?=(Objetivos|Contenido|Actividad|Ejemplo|Resumen|Evaluación)\b)", "\n\n", text, flags=re.IGNORECASE)

    # Limpia "vamos a ver", "básicamente" repetido
    text = re.sub(r"\b(básicamente|en realidad|literalmente)\b\s*,?\s*\b\1\b", r"\1", text, flags=re.IGNORECASE)

    if text != before:
        report.add("Aplicado perfil DOCENTE (claridad y estructura ligera).")
    return text


def apply_profile_linkedin(text: str, report: FixReport) -> str:
    """
    LinkedIn: gancho + párrafos cortos + ritmo + CTA suave (opcional, no agresivo).
    No inventa datos; solo reestructura si el texto es un bloque.
    """
    before = text
    t = text.strip()

    # Si parece un bloque enorme sin saltos, crea respiración.
    if "\n" not in t and len(t) > 400:
        # divide por frases
        t = re.sub(r"\.\s+", ".\n\n", t)

    # Reduce frases muy formales
    t = re.sub(r"\bAsimismo\b", "También", t, flags=re.IGNORECASE)
    t = re.sub(r"\bEn consecuencia\b", "Por eso", t, flags=re.IGNORECASE)

    # Evita cierre robot tipo "En conclusión"
    t = re.sub(r"(?i)\bEn conclusión\b\s*[:,]?\s*", "", t)

    # CTA suave si no existe un cierre claro
    if len(t) > 250 and not re.search(r"(?i)\b(¿qué opinas|te leo|opiniones|comentarios)\b", t):
        t += "\n\n¿Te ha pasado algo parecido? Te leo 👇"

    text = t
    if text != before:
        report.add("Aplicado perfil LINKEDIN (párrafos cortos + ritmo + CTA suave).")
    return text


def apply_profile_helpdesk(text: str, report: FixReport) -> str:
    """
    Helpdesk: formato claro tipo ticket:
    - Síntoma / Causa probable / Pasos / Verificación / Escalado
    Si el texto ya tiene pasos, los respeta.
    """
    before = text
    t = text.strip()

    # Si no tiene secciones y es largo, añade headings sin inventar contenido,
    # simplemente re-encuadra.
    has_headings = bool(re.search(r"(?im)^(síntoma|causa|pasos|solución|verificación|notas|escalado)\b", t))
    has_steps = bool(re.search(r"(?m)^\s*(\d+[\).]|- |\* )", t))

    if not has_headings and len(t) > 300:
        # Hace una estructura mínima
        t = (
            "Síntoma:\n"
            f"{t}\n\n"
            "Pasos recomendados:\n"
            "- Reproducir el problema y recopilar detalles (mensaje exacto, hora, capturas).\n"
            "- Revisar logs/eventos relacionados.\n"
            "- Aplicar fix mínimo y validar.\n\n"
            "Verificación:\n"
            "- Confirmar con el usuario que el fallo no reaparece.\n\n"
            "Notas / Escalado:\n"
            "- Si persiste, escalar con logs y pasos ya realizados."
        )
    elif has_steps:
        # Limpia numeración inconsistente: "1)" "1." -> "1."
        t = re.sub(r"(?m)^\s*(\d+)\)\s*", r"\1. ", t)
        t = re.sub(r"(?m)^\s*(\d+)\.\s*", r"\1. ", t)

    text = t
    if text != before:
        report.add("Aplicado perfil HELPDESK (formato ticket + pasos claros).")
    return text


def apply_profile(text: str, profile: str, report: FixReport) -> str:
    profile = profile.lower().strip()
    if profile == "neutro":
        return apply_profile_neutro(text, report)
    if profile == "docente":
        return apply_profile_docente(text, report)
    if profile == "linkedin":
        return apply_profile_linkedin(text, report)
    if profile == "helpdesk":
        return apply_profile_helpdesk(text, report)
    raise ValueError(f"Perfil desconocido: {profile}")


# -----------------------------
# IA Check Control
# -----------------------------
def _sentence_lengths(text: str) -> List[int]:
    # Segmentación simple por puntuación fuerte.
    chunks = re.split(r"[.!?]+\s+", text.strip())
    return [len(c.strip()) for c in chunks if c.strip()]


def ai_check_control(text: str, report: FixReport) -> None:
    """
    Genera un score 0-100 (más alto = más "huele a IA").
    Heurístico, no infalible.
    """
    t = text.strip()
    if not t:
        report.ai_score = 0
        return

    # métricas
    lengths = _sentence_lengths(t)
    avg_len = sum(lengths) / max(1, len(lengths))
    long_sent_ratio = sum(1 for x in lengths if x > 220) / max(1, len(lengths))

    # conectores frecuentes
    connectors = re.findall(r"\b(además|sin embargo|no obstante|por otro lado|en este sentido)\b", t, flags=re.IGNORECASE)
    conn_rate = len(connectors) / max(1, len(t.split()))

    # muletillas IA
    tic_hits = 0
    for rx in AI_TICS_REGEX:
        tic_hits += len(re.findall(rx, t, flags=re.IGNORECASE))

    # boilerplate (autodeclaraciones)
    boiler_hits = 0
    for rx in BOILERPLATE_REGEX:
        boiler_hits += len(re.findall(rx, t, flags=re.IGNORECASE))

    # repetición simple (palabras top)
    words = [w.lower() for w in re.findall(r"[\wáéíóúüñ]+", t)]
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_rep = sum(v for _, v in top) / max(1, len(words))

    # score heurístico
    score = 0.0
    score += min(25.0, (avg_len - 90) * 0.25) if avg_len > 90 else 0.0
    score += long_sent_ratio * 25.0
    score += min(20.0, tic_hits * 2.0)
    score += min(15.0, boiler_hits * 5.0)
    score += min(15.0, max(0.0, (top_rep - 0.25)) * 60.0)  # si demasiada repetición de palabras

    # normaliza
    score_int = int(max(0, min(100, round(score))))
    report.ai_score = score_int

    # flags
    if avg_len > 160:
        report.add_flag(f"Frases largas (promedio {avg_len:.0f} chars).")
    if long_sent_ratio > 0.20:
        report.add_flag(f"Muchas frases >220 chars ({long_sent_ratio*100:.0f}%).")
    if tic_hits >= 5:
        report.add_flag(f"Muletillas/conectores típicos de IA detectados: {tic_hits}.")
    if boiler_hits >= 1:
        report.add_flag(f"Boilerplate/autodeclaraciones detectadas: {boiler_hits}.")
    if conn_rate > 0.04:
        report.add_flag(f"Exceso de conectores (ratio {conn_rate:.3f}).")
    if top_rep > 0.35:
        report.add_flag(f"Repetición alta de vocabulario (top10 ratio {top_rep:.2f}).")

    report.metrics.update({
        "sentences": float(len(lengths)),
        "avg_sentence_len_chars": float(avg_len),
        "long_sentence_ratio": float(long_sent_ratio),
        "connector_rate": float(conn_rate),
        "tic_hits": float(tic_hits),
        "boilerplate_hits": float(boiler_hits),
        "top10_word_ratio": float(top_rep),
    })


# -----------------------------
# LanguageTool (opcional)
# -----------------------------
def languagetool_correct(text: str, report: FixReport, lang: str = "es") -> str:
    try:
        import language_tool_python  # type: ignore
    except Exception:
        report.add("LanguageTool no está instalado: omitida corrección gramatical avanzada.")
        return text

    before = text
    tool = language_tool_python.LanguageTool(lang)
    text = tool.correct(text)
    if text != before:
        report.add(f"Aplicada corrección gramatical avanzada (LanguageTool: {lang}).")
    else:
        report.add("LanguageTool: no se detectaron cambios relevantes.")
    return text


# -----------------------------
# Diff simple (opcional)
# -----------------------------
def simple_diff(a: str, b: str) -> str:
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    out = ["=== Diff (simple) ==="]
    max_len = max(len(a_lines), len(b_lines))
    changes = 0
    for i in range(max_len):
        old = a_lines[i] if i < len(a_lines) else ""
        new = b_lines[i] if i < len(b_lines) else ""
        if old != new:
            changes += 1
            out.append(f"\nLínea {i+1}:")
            out.append(f"- {old}")
            out.append(f"+ {new}")
    if changes == 0:
        out.append("\n(Sin diferencias)")
    return "\n".join(out)


# -----------------------------
# Pipeline principal
# -----------------------------
def fix_text(
    text: str,
    report: FixReport,
    profile: str = "neutro",
    use_languagetool: bool = False,
    lt_lang: str = "es",
    do_ai_check: bool = True,
    split_sentences_enabled: bool = True,
) -> str:
    report.original_len = len(text)

    # 1) Normalizaciones técnicas
    text = normalize_quotes(text, report)
    text = normalize_whitespace(text, report)
    text = normalize_punctuation(text, report)

    # 2) Limpieza IA/estilo base
    text = reduce_ai_tics(text, report)
    text = dedupe_repetitions(text, report)
    if split_sentences_enabled:
        text = split_long_sentences(text, report)

    # 3) Perfil
    text = apply_profile(text, profile, report)

    # 4) (Opcional) Corrección gramatical avanzada
    if use_languagetool:
        text = languagetool_correct(text, report, lt_lang)

    # 5) Último pase
    text = normalize_whitespace(text, report)
    report.final_len = len(text)

    # 6) IA Check Control (al final, para medir output)
    if do_ai_check:
        ai_check_control(text, report)

    return text


# -----------------------------
# CLI
# -----------------------------
def read_input(args: argparse.Namespace) -> str:
    if args.stdin:
        return sys.stdin.read()
    if not args.input:
        raise SystemExit("ERROR: Debes usar --in <archivo> o --stdin.")
    with open(args.input, "r", encoding="utf-8") as f:
        return f.read()


def write_output(args: argparse.Namespace, text: str) -> None:
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AI Text Fixer PRO (ES) — perfiles Docente/LinkedIn/Helpdesk + IA Check Control."
    )
    p.add_argument("--in", dest="input", help="Archivo de entrada .txt/.md", default=None)
    p.add_argument("--out", dest="output", help="Archivo de salida", default=None)
    p.add_argument("--stdin", action="store_true", help="Lee el texto desde stdin")
    p.add_argument("--profile", choices=["neutro", "docente", "linkedin", "helpdesk"], default="neutro",
                   help="Perfil de estilo")
    p.add_argument("--report", action="store_true", help="Imprime reporte por stderr")
    p.add_argument("--diff", action="store_true", help="Imprime diff simple por stderr")
    p.add_argument("--ai-check", action="store_true", help="Activa IA Check Control (score + alertas)")
    p.add_argument("--no-ai-check", action="store_true", help="Desactiva IA Check Control")
    p.add_argument("--no-split", action="store_true", help="No dividir frases largas")
    p.add_argument("--languagetool", metavar="LANG", nargs="?", const="es", default=None,
                   help="Activa corrección con LanguageTool (ej: es, es-ES). Requiere: pip install language-tool-python")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    original = read_input(args)
    report = FixReport()

    use_lt = args.languagetool is not None
    lt_lang = args.languagetool if args.languagetool else "es"

    do_ai_check = args.ai_check and not args.no_ai_check

    fixed = fix_text(
        original,
        report,
        profile=args.profile,
        use_languagetool=use_lt,
        lt_lang=lt_lang,
        do_ai_check=do_ai_check,
        split_sentences_enabled=not args.no_split,
    )

    write_output(args, fixed)

    if args.diff:
        sys.stderr.write("\n" + simple_diff(original, fixed) + "\n")

    if args.report:
        sys.stderr.write("\n" + report.render() + "\n")


if __name__ == "__main__":
    main()
