# Sistema de Recomendación Multi-LLM e Híbrido

Este proyecto implementa un motor de recomendación avanzado para *Serious Games*, utilizando una arquitectura de RAG (Retrieval-Augmented Generation), múltiples proveedores de LLMs (OpenAI, Anthropic, Google), Filtrado Colaborativo Distribuido (FCD) y diversificación MMR.

## Estructura del Proyecto

- `src/`: Módulos de lógica del sistema.
  - `config.py`: Configuraciones globales y constantes.
  - `data_loader.py`: Gestión de carga de datasets CSV.
  - `llm_clients.py`: Abstracción de proveedores de lenguaje.
  - `vector_db.py`: Integración con ChromaDB y embeddings.
  - `algorithms.py`: Algoritmos de ranking, FCD y MMR.
  - `recommender.py`: Orquestador principal de recomendaciones.
- `data/dataset/`: Almacén de archivos CSV fuente.
- `chroma_db/`: Base de datos vectorial persistente.
- `main.py`: Punto de entrada principal.

## Configuración del Entorno

Sigue estos pasos para configurar tu entorno local:

1. **Crear entorno virtual**:
   ```bash
   python -m venv venv
   ```

2. **Activar el entorno**:
   - Mac/Linux:
     ```bash
     source venv/bin/activate
     ```
   - Windows:
     ```bash
     venv\Scripts\activate
     ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno**:
   - Copia el archivo `.env.example` a uno nuevo llamado `.env`.
   - Edita el archivo `.env` y añade tus claves de API para los proveedores que desees usar (OpenAI, Anthropic, Gemini, Groq).

## Uso

Para ejecutar una recomendación de ejemplo:
```bash
python main.py "tu consulta de búsqueda aquí"
```

Si no se proporciona una consulta, el sistema usará una por defecto relacionada con toma de decisiones y trabajo en equipo.

## Características Principales

- **Multi-LLM**: Consulta simultánea a varios modelos para obtener consenso.
- **Validación Semántica**: Los resultados de los LLMs se validan contra la base de datos real.
- **Ranking Híbrido**: Combina relevancia semántica con preferencias personales (FCD).
- **Diversidad (MMR)**: Asegura que los resultados no sean redundantes.
- **Optimizado**: Usa RapidFuzz y operaciones vectorizadas para mayor velocidad.
