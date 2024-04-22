# Implementación de Cadena de LangChain con Streamlit

Este repositorio permite implementar una cadena de [LangChain] con [Streamlit] para chatear con el contenido de un website


## Requerimientos
- [LangChain library](https://python.langchain.com/en/latest/index.html)
- [OpenAI API key](https://platform.openai.com/)

## Installation

#### 1. Clonar el repositorio

```bash
git clone https://github.com/sergiovogt/agent-langchain-website.git
```

#### 2. Crear el entorno

``` bash
cd agent-langchain-website
python -m venv env
source env/bin/activate
```

#### 3. Instalar las dependencias requeridas
``` bash
pip install -r requirements.txt
```

Primero, crear el archvio `.env` en el directorio raiz del proyecto. Dentro del archvio, agreagar la API Key de OpenAI:

```makefile
OPENAI_API_KEY="agregar_aquí_la_apikey"
```

Guardar el archivo y cerrarlo. En el script de Python, cargar el archivo `.env` usando el siguiente código (ya está cargado en [backend.py]):
```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
```

Ahora tu entorno de Python está listo! ya puedes continuar...


## Tutoriales
Para ver más tutoriales, podés visitar mi canal de YouTube:  [youtube.com/@sergiovogtds1998](https://youtube.com/@sergiovogtds1998)
