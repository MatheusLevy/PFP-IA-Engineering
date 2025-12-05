# Repositório de Código de Engenharia em Machine Learning (MLOps)
## Descrição
Este repositório funciona como um **ambiente de produção para implantação e gerenciamento de modelos de Machine Learning**, transformando notebooks de pesquisa em código Python executável e otimizado. Ele automatiza o fluxo de trabalho desde a preparação de dados até a validação e implantação de modelos em ambientes de produção, garantindo robustez, escalabilidade e monitoramento contínuo.

### Principais Funcionalidades
*   **Transformação de Notebooks em Código Produtivo**: Converte notebooks Jupyter de pesquisa em scripts Python modulares (`steps/`).
*   **Pipeline de Implantação Automatizado**: Orquestra sequência de passos (preparação de dados, treinamento, registro, promoção) via scripts integrados.
*   **Versionamento e Rastreabilidade**: Integra DVC para dados e MLflow para modelos, garantindo reprodução exata de experimentos.
*   **Validação Automática de Modelos**: Compara desempenho entre modelos em staging e produção para decisões de implantação.
*   **Gestão de Ciclo de Vida**: Gerencia promoção entre estágios (Staging → Production) com critérios de desempenho.

### Público-Alvo Primário
*   **Engenheiros de Machine Learning/MLOps**: Profissionais responsáveis por implantar e manter modelos em produção.
*   **Desenvolvedores de Software**: Integradores que incorporam modelos ML em aplicações maiores.
*   **Equipes de DevOps**: Responsáveis por infraestrutura e pipelines de CI/CD.

### Natureza do Projeto
**Módulo de Sistema de Produção** (parte da suíte PFP) que opera como um **MVP para ambientes de engenharia**, demonstrando práticas de MLOps em escala controlada com:
*   Componentes modulares e testáveis
*   Integração com ferramentas industriais (MLflow, DVC)
*   Transição gradual entre pesquisa e produção

### Ressalvas Importantes
1.  **Configuração de Infraestrutura**: Requer serviços MLflow e Minio/S3 pré-configurados.
2.  **Monitoramento Limitado**: Não inclui ferramentas avançadas de monitoramento de modelos em produção.
3.  **Escalabilidade**: Projetado para cargas de trabalho moderadas; requer ajustes para escala industrial.

---

## Visão de Projeto
### Cenário Positivo 1: Implantação Contínua de Novas Versões
**Persona:** Carla, Engenheira de MLOps.
**Contexto:** Carla precisa implantar uma nova versão do modelo YOLO otimizado pela equipe de pesquisa, garantindo que ele supere o modelo atual em produção sem interromper o serviço.

**Narrativa:** Carla executa o pipeline (`steps/01_generate_splits.py` → `05_validate_with_production.py`) que automaticamente:
1.  Prepara novos splits de dados
2.  Treina e registra o modelo no MLflow
3.  Promove para staging após validação básica
4.  Compara com o modelo em produção
5.  Implanta somente se houver melhoria ≥1% em mAP50

O processo termina com o novo modelo em produção e métricas registradas no MLflow, sem intervenção manual.

### Cenário Positivo 2: Rollback Automático
**Persona:** Diego, DevOps Engineer.
**Contexto:** Um novo modelo em produção causa degradação de performance não detectada nos testes iniciais.

**Narrativa:** O sistema de monitoramento detecta queda no mAP50 via API do MLflow. Diego executa `05_validate_with_production.py` forçando comparação com a versão anterior. O script identifica a regressão e automaticamente:
1.  Reverte para o modelo estável
2.  Atualiza tags no MLflow
3.  Gera alerta para a equipe de pesquisa

### Cenário Negativo: Falha na Compatibilidade de Dependências
**Persona:** Carla, Engenheira de MLOps.
**Contexto:** Um modelo treinado com nova versão do PyTorch falha ao ser implantado no ambiente de produção desatualizado.

**Narrativa:** O script `03_log_model.py` registra o modelo com sucesso, mas o contêiner de produção falha ao carregá-lo. Carla gasta horas rastreando incompatibilidades entre versões de bibliotecas, percebendo que o sistema não verifica automaticamente compatibilidade runtime durante o registro.

### Cenário Negativo: Descompasso Entre Splits de Dados
**Persona:** Eduardo, Cientista de Dados.
**Contexto:** Alterações na preparação de dados causam inconsistências entre splits de treino/validação usados no treinamento e produção.

**Narrativa:** Eduardo atualiza `01_generate_splits.py` mas esquece de reexecutar o pipeline completo. O modelo é treinado com dados novos mas validado com splits antigos, causando superestimação de performance. O erro só é detectado após implantação.

---

## Documentação Técnica
### 1. Especificação de Requisitos
#### Funcionais:
| ID     | Descrição                                                                 |
|--------|---------------------------------------------------------------------------|
| RF01   | Transformar notebooks de pesquisa em scripts Python modulares              |
| RF02   | Executar pipeline sequencial (preparação → treinamento → implantação)     |
| RF03   | Validar modelos automaticamente contra versão de produção                  |
| RF04   | Gerenciar estágios de modelos (Staging/Production) via tags MLflow      |
| RF05   | Garantir reprodutibilidade com versionamento de dados (DVC) e código (Git) |

#### Não-Funcionais:
| ID     | Descrição                                                     |
|--------|---------------------------------------------------------------|
| RNF01  | Tempo máximo de 2h para execução completa do pipeline         |
| RNF02  | Compatibilidade com Python 3.14+                              |
| RNF03  | Logs detalhados em cada etapa do processo                    |
| RNF04  | Tolerância a falhas com rollback automático                    |
| RNF05  | Configuração via variáveis de ambiente                        |

### 2. Arquitetura e Fluxo de Dados
```
[Repositório de Pesquisa]
          ↓
[Scripts de Engenharia (steps/)] → [Dados Preparados (dataset/)]
          ↓
[Modelo Treinado (MLflow)] → [Validação] → [Modelo em Produção]
          ↑
[Configuração (docker-compose.yaml)]
```

#### Estrutura de Diretórios:
```
.
├── steps/                   # Scripts do pipeline
│   ├── 01_generate_splits.py   # Preparação de dados
│   ├── 02_train_model.py       # Treinamento
│   ├── 03_log_model.py         # Registro no MLflow
│   ├── 04_promote_to_staging.py# Promoção para staging
│   └── 05_validate_with_production.py # Validação final
├── dataset/                 # Dados processados
├── data/                    # Dados brutos versionados (DVC)
├── dvc.lock                 # Estado atual dos dados
└── .env                     # Configurações de ambiente
```

### 3. Fluxos de Trabalho
#### Pipeline Principal:
1.  **Preparação de Dados** (`01_generate_splits.py`):
    ```bash
    python steps/01_generate_splits.py
    ```
    - Lê `dataset_info.csv`
    - Gera splits (train/valid/test)
    - Cria arquivos YAML de configuração

2.  **Treinamento** (`02_train_model.py`):
    ```bash
    python steps/02_train_model.py
    ```
    - Otimiza hiperparâmetros com Optuna
    - Registra métricas no Weights & Biases
    - Salva melhor modelo em `runs/detect/`

3.  **Registro** (`03_log_model.py`):
    ```bash
    python steps/03_log_model.py
    ```
    - Valida desempenho mínimo
    - Registra modelo no MLflow
    - Atualiza `training_info.json`

4.  **Promoção para Staging** (`04_promote_to_staging.py`):
    ```bash
    python steps/04_promote_to_staging.py
    ```
    - Verifica requisitos de performance
    - Atualiza tags no MLflow
    - Gera novo versionamento

5.  **Validação Final** (`05_validate_with_production.py`):
    ```bash
    python steps/05_validate_with_production.py
    ```
    - Compara com modelo em produção
    - Promove para produção se superior
    - Reverte em caso de regressão

### 4. Configuração Chave
#### Variáveis de Ambiente (`.env`):
```ini
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://localhost:9444
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
```

#### Dependências Principais (`pyproject.toml`):
```toml
[project]
dependencies = [
  "mlflow>=2.8.1",
  "optuna>=3.4.0",
  "ultralytics>=8.0.0",
  "pandas>=2.0.0",
  "torch>=2.0.0"
]
```

---

## Manual de Utilização
### 1. Configuração Inicial
```bash
git clone https://github.com/seu-usuario/pfp-ia-engineering.git
cd pfp-ia-engineering
uv sync  # Instala dependências
cp .env.example .env  # Configura ambiente
```

### 2. Execução do Pipeline Completo
```bash
# Passo 1: Preparar dados
python steps/01_generate_splits.py

# Passo 2: Treinar modelo
python steps/02_train_model.py

# Passo 3: Registrar no MLflow
python steps/03_log_model.py

# Passo 4: Promover para staging
python steps/04_promote_to_staging.py

# Passo 5: Validar e implantar
python steps/05_validate_with_production.py
```

### 3. Monitoramento e Validação
**Verificar status no MLflow:**
```bash
open http://localhost:5000  # Acessar UI
```

**Consultar tags de estágio:**
```python
from mlflow import MlflowClient
client = MlflowClient()
version = client.get_model_version(name="yolo-rock-paper-scissors", version=1)
print(version.tags['stage'])  # 'production' ou 'staging'
```

### 4. Gerenciamento de Erros Comuns
| Problema                           | Solução                                  |
|------------------------------------|------------------------------------------|
| Falha na conexão com MLflow        | Verificar `docker-compose up` e `.env`   |
| Modelo não atinge métricas mínimas | Ajustar limiares em `04_promote_to_staging.py` |
| Incompatibilidade de dependências  | Executar `uv sync --force-reinstall`     |
| DVC desatualizado                  | Executar `dvc pull`                      |

---