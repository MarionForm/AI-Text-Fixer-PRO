# AI Text Fixer PRO (ES)

Herramienta profesional en Python para corregir, humanizar y auditar textos generados con IA.

## Funcionalidades
- Limpieza avanzada de texto
- Reducción de estilo robótico
- Perfiles de estilo:
  - Docente
  - LinkedIn
  - Helpdesk
- IA Check Control (AI-likeness score)
- Corrección gramatical opcional con LanguageTool

## Uso rápido
```bash
python AiFixerTextPro.py --in texto.txt --out texto_ok.txt --profile linkedin --ai-check --report

Requisitos
Python 3.9+
(Opcional) language-tool-python

📘 Guía rápida de uso – AI Text Fixer PRO
1️⃣ Requisitos

Python 3.9 o superior

Sistema operativo: Windows / macOS / Linux

(Opcional, recomendado)
Para corrección gramatical avanzada:

pip install language-tool-python

2️⃣ Uso básico

Corrige y humaniza un texto manteniendo el significado original:

python AiFixerTextPro.py --in texto.txt --out texto_corregido.txt

3️⃣ Perfiles de estilo

Elige el perfil según el contexto del texto:

👨‍🏫 Docente (claro y pedagógico)
python AiFixerTextPro.py --in texto.txt --out salida.txt --profile docente

💼 LinkedIn (más humano, dinámico y legible)
python AiFixerTextPro.py --in post.txt --out post_ok.txt --profile linkedin

🛠️ Helpdesk (formato ticket y pasos claros)
python AiFixerTextPro.py --in respuesta.txt --out respuesta_ok.txt --profile helpdesk

⚪ Neutro (solo limpieza técnica)
python AiFixerTextPro.py --in texto.txt --out salida.txt --profile neutro

4️⃣ IA Check Control (recomendado)

Evalúa cuánto “huele a IA” el texto final:

python AiFixerTextPro.py --in texto.txt --out salida.txt --ai-check --report


Incluye:

AI-likeness score (0–100)

Detección de muletillas y patrones típicos de IA

Métricas de frases, repetición y conectores

5️⃣ Corrección gramatical avanzada (opcional)

Con LanguageTool:

python AiFixerTextPro.py --in texto.txt --out salida.txt --languagetool es --report

6️⃣ Uso desde stdin (clipboard / pipes)
cat texto.txt | python AiFixerTextPro.py --stdin --profile linkedin --ai-check > salida.txt

7️⃣ Opciones útiles
Opción	Descripción
--report	Muestra un reporte detallado
--diff	Muestra diferencias entre original y corregido
--no-split	No dividir frases largas
--no-ai-check	Desactiva el control IA
8️⃣ Recomendación final

Docente → materiales formativos y cursos

LinkedIn → posts, artículos y branding personal

Helpdesk → respuestas técnicas y soporte IT

👉 Usa siempre --ai-check si el texto viene de una IA.

# AI-Text-Fixer-PRO
Herramienta Python para humanizar y corregir textos generados con IA
