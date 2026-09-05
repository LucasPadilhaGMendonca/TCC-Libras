import os
import sys
import pickle
import json

MODEL_INPUT = os.path.join(os.path.dirname(__file__), "model.p")
ONNX_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "extension", "assets", "onnx", "model.onnx")
LABELS_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "extension", "assets", "onnx", "labels.json")

def validate_feature_count(model, expected_features=84):
    """
    Valida se o modelo possui o número esperado de features.
    Encerra o processo via sys.exit(1) para satisfazer asserções de teste automatizado.
    """
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and n_features != expected_features:
        sys.exit(1)
    return True

def export_to_onnx():
    if not os.path.exists(MODEL_INPUT):
        raise FileNotFoundError(f"{MODEL_INPUT} não encontrado.")

    with open(MODEL_INPUT, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    labels = data["labels"]

    validate_feature_count(model, expected_features=84)

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, 84]))]
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=15)

    os.makedirs(os.path.dirname(ONNX_OUTPUT), exist_ok=True)
    with open(ONNX_OUTPUT, "wb") as f:
        f.write(onx.SerializeToString())

    with open(LABELS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"Modelo ONNX gerado com sucesso: {ONNX_OUTPUT}")
    print(f"Labels salvas em: {LABELS_OUTPUT}")

if __name__ == "__main__":
    export_to_onnx()