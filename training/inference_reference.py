# Referência Python do pré-processamento usado pela extensão.
# Não é usado pela extensão; serve para validar que o vetor possui 42 features.

def build_features(hand_landmarks):
    data_aux = []
    x_ = []
    y_ = []

    for landmark in hand_landmarks:
        x_.append(landmark.x)
        y_.append(landmark.y)

    for landmark in hand_landmarks:
        data_aux.append(landmark.x - min(x_))
        data_aux.append(landmark.y - min(y_))

    if len(data_aux) != 42:
        return None

    return data_aux
