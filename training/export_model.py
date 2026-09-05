import os
import json
import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

MODEL_INPUT = "training/model.p"
ONNX_OUTPUT = "extension/assets/onnx/model.onnx"
LABELS_OUTPUT = "extension/assets/onnx/labels.json"

if not os.path.exists(MODEL_INPUT):
    raise FileNotFoundError(f"{MODEL_INPUT} não encontrado.")

with open(MODEL_INPUT, "rb") as f:
    payload = pickle.load(f)

model = payload["model"]

# 84 features (bimanual: 2 mãos x 21 pontos x 2 eixos)
initial_type = [("input", FloatTensorType([None, 84]))]

print("Convertendo modelo para ONNX (84 features, IR Version <= 8 / target_opset=15)...")

# target_opset=15 garante compatibilidade com o runtime web da extensão
onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    target_opset=15,
    options={id(model): {"zipmap": False}}
)

# Força ir_version = 8 caso a biblioteca tente gravar 9 ou 10
onnx_model.ir_version = 8

os.makedirs(os.path.dirname(ONNX_OUTPUT), exist_ok=True)
with open(ONNX_OUTPUT, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Modelo ONNX gerado com sucesso: {os.path.abspath(ONNX_OUTPUT)}")

# Exportação do mapeamento de classes
labels_list = [str(cls) for cls in model.classes_]
with open(LABELS_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(labels_list, f, ensure_ascii=False, indent=2)

print(f"Labels salvas em: {os.path.abspath(LABELS_OUTPUT)}")