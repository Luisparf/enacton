# Enacton  – Música Interativa com IA e Cognição 4E

Este projeto explora a **cognição 4E** (Embodied, Embedded, Enactive, Extended) aplicada à música interativa.  
A ideia central: **gestos corporais do performer não são apenas movimento, mas cognição em ato**.  
Com uma câmera e IA, transformamos esses gestos em controle sonoro em tempo real.

---

## Como funciona

Pipeline do sistema:


- **Embodied** → Captura de corpo/mãos como parte do pensamento musical.  
- **Embedded** → Contexto (instrumento, seção da peça, palco) modula a interpretação.  
- **Enactive** → Performer e IA co-criam o fluxo.  
- **Extended** → A IA age como extensão cognitiva do músico.  

---

## Instalação

### 1. Criar ambiente virtual
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
echo "$VIRTUAL_ENV"



## Execução 

# ex:
python -m mpfs.run_face --cam 0 --style boxes_brows --line-thick 1 --alpha 0.70
python -m mpfs.run_face --cam 0 -di -ee
# ou
python run_face.py --style hud

# tudo: pontos coloridos + íris + ROIs/barras
python -m mpfs.run_micro --preview --show-all

# só nuvem completa (sem ROIs/barras)
python -m mpfs.run_micro --preview --show-all --no-roi

# depuração com IDs (cuidado: fica carregado)
python -m mpfs.run_micro --preview --show-all --show-ids


# mesh
python -m mpfs.run_micro --preview --mesh mp 

python -m mpfs.run_micro --preview --mesh mp --show-all --show-id --events
