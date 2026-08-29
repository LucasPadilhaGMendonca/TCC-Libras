import os
import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(ROOT, "training", "model.p")
OUT_FILE = os.path.join(ROOT, "extension", "assets", "onnx", "model.onnx")


def validate_feature_count(n_features, expected=42):
    if n_features != expected:
        raise SystemExit(
            f"O modelo possui {n_features} features. "
            f"A extensão foi preparada para {expected}."
        )


def main():
    if not os.path.exists(MODEL_FILE):
        raise SystemExit("training/model.p não encontrado. Execute train_classifier.py primeiro.")

    with open(MODEL_FILE, "rb") as f:
        model_dict = pickle.load(f)

    model = model_dict["model"]
    validate_feature_count(model.n_features_in_)

    initial_type = [("float_input", FloatTensorType([None, 42]))]

    # zipmap=False facilita o consumo da saída de probabilidades no navegador.
    options = {id(model): {"zipmap": False}}

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        options=options,
        target_opset=17
    )

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    with open(OUT_FILE, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Modelo ONNX gerado: {OUT_FILE}")


if __name__ == "__main__":
    main()
