# Sistema de Recomendación Multi-LLM e Híbrido para Serious Games

Este proyecto implementa un motor de recomendación avanzado para *Serious Games*, utilizando una arquitectura de **RAG (Retrieval-Augmented Generation)**, múltiples proveedores de LLMs (OpenAI, Anthropic, Google), Filtrado Colaborativo Distribuido (FCD) y diversificación MMR.

El sistema no solo recomienda juegos existentes, sino que sintetiza una **propuesta de diseño integrada** basada en las mejores mecánicas, dinámicas y narrativas analizadas.

## 🚀 Características Principales

- **Análisis Multi-Perspectiva (EDNM)**: Evalúa recomendaciones desde 4 ejes clave: Elementos, Dinámicas, Narrativas y Mecánicas.
- **Consenso Multi-LLM**: Consulta simultánea a modelos de vanguardia (GPT-4o, Claude 3.5, Gemini Pro) para obtener un consenso robusto.
- **Validación Semántica**: Los resultados generados por los LLMs se validan contra una base de datos real en ChromaDB utilizando embeddings de HuggingFace.
- **Síntesis Creativa**: Genera una propuesta de diseño narrativa y técnica, justificando cada componente desde una perspectiva pedagógica y científica.
- **Métricas de Calidad**: Implementación de **BERTScore** para evaluar la calidad y coherencia de las salidas generadas por los distintos LLMs.
- **Interfaz Web Moderna**: Dashboard interactivo para realizar consultas y visualizar propuestas de diseño en tiempo real.

## 📁 Estructura del Proyecto

- `src/`: Módulos de lógica del sistema.
  - `recommender.py`: Orquestador principal (EDNM + Síntesis).
  - `llm_clients.py`: Abstracción de proveedores (OpenAI, Anthropic, Google).
  - `vector_db.py`: Integración con ChromaDB y embeddings.
  - `evaluation.py`: Implementación de métricas BERTScore.
  - `config.py`: Configuraciones globales.
- `app.py`: Servidor Flask para la API y la interfaz web.
- `main.py`: Punto de entrada para ejecución en terminal (CLI).
- `stitch/`: Sistema de diseño para el frontend.
- `data/`: Datasets de juegos y ratings.

## 🛠️ Configuración e Instalación

1. **Clonar el repositorio y crear entorno virtual**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   # .venv\Scripts\activate  # Windows
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Variables de Envorno**:
   Crea un archivo `.env` basado en `.env.example` con tus API Keys:
   ```env
   OPENAI_API_KEY=tu_clave
   ANTHROPIC_API_KEY=tu_clave
   GOOGLE_API_KEY=tu_clave
   ```

## 💻 Modos de Uso

### Interfaz Web (Recomendado)
Para iniciar el servidor interactivo:
```bash
PYTHONPATH=. python app.py
```
Accede a `http://localhost:5001` en tu navegador.

### Línea de Comandos (CLI)
Para ejecutar una recomendación rápida:
```bash
PYTHONPATH=. python main.py --query "tu consulta aquí" --verbose
```

## 🧪 Evaluación de Calidad
El sistema utiliza **BERTScore** para comparar las salidas de los modelos contra una referencia experta o entre sí, permitiendo identificar qué proveedor de LLM ofrece la propuesta más coherente para cada tipo de consulta.

---
*Desarrollado para la optimización del diseño de Serious Games mediante IA Generativa.*
