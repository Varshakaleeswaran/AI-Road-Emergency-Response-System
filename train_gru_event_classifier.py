import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, classification_report

DATA = r"data\ACCIDENT\gru_sequences.npz"

data = np.load(DATA, allow_pickle=True)

X = data["X"]
y = data["y"]
groups = data["groups"]

print("=" * 60)
print("GRU EVENT CLASSIFIER")
print("=" * 60)
print("Sequences:", len(X))
print("Shape:", X.shape)
print("Videos:", len(np.unique(groups)))
print()

class GRUClassifier(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        output, hidden = self.gru(x)

        last = output[:, -1, :]

        last = self.dropout(last)

        return self.fc(last).squeeze(1)


device = torch.device("cpu")

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_results = []

all_true = []
all_pred = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    X_train = X[train_idx].copy()
    X_test = X[test_idx].copy()

    y_train = y[train_idx]
    y_test = y[test_idx]

    # Fit scaler ONLY on training data
    scaler = StandardScaler()

    train_2d = X_train.reshape(-1, X_train.shape[-1])

    scaler.fit(train_2d)

    X_train = scaler.transform(
        train_2d
    ).reshape(X_train.shape)

    X_test = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)

    # Replace any remaining NaN values
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32,
        device=device
    )

    y_train_t = torch.tensor(
        y_train,
        dtype=torch.float32,
        device=device
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32,
        device=device
    )

    model = GRUClassifier(
        input_size=X.shape[2]
    ).to(device)

    # Handle class imbalance
    positive = max(np.sum(y_train == 1), 1)
    negative = max(np.sum(y_train == 0), 1)

    pos_weight = torch.tensor(
        [negative / positive],
        dtype=torch.float32,
        device=device
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    model.train()

    for epoch in range(100):

        optimizer.zero_grad()

        logits = model(X_train)

        loss = criterion(
            logits,
            y_train_t
        )

        loss.backward()

        optimizer.step()

    model.eval()

    with torch.no_grad():

        logits = model(X_test)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).cpu().numpy().astype(int)

    bal_acc = balanced_accuracy_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    fold_results.append(
        (bal_acc, f1)
    )

    all_true.extend(y_test.tolist())
    all_pred.extend(predictions.tolist())

    print(
        f"Fold {fold}: "
        f"balanced_acc={bal_acc:.3f} "
        f"F1={f1:.3f}"
    )


print()
print("=" * 60)
print("GRU RESULTS")
print("=" * 60)

mean_bal = np.mean(
    [x[0] for x in fold_results]
)

std_bal = np.std(
    [x[0] for x in fold_results]
)

mean_f1 = np.mean(
    [x[1] for x in fold_results]
)

std_f1 = np.std(
    [x[1] for x in fold_results]
)

print(
    f"Mean balanced accuracy: "
    f"{mean_bal:.4f}"
)

print(
    f"Std balanced accuracy: "
    f"{std_bal:.4f}"
)

print(
    f"Mean F1: "
    f"{mean_f1:.4f}"
)

print(
    f"Std F1: "
    f"{std_f1:.4f}"
)

print()
print("=" * 60)
print("OUT-OF-FOLD CONFUSION MATRIX")
print("=" * 60)

print(
    confusion_matrix(
        all_true,
        all_pred
    )
)

print()
print(
    classification_report(
        all_true,
        all_pred,
        target_names=[
            "non_event",
            "accident_event"
        ],
        zero_division=0
    )
)
