import random
import np
import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet18, ResNet18_Weights
from skorch import NeuralNetClassifier
from pytorch_optimizer import Lookahead
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
import matplotlib.pyplot as plt

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Call fuction from data.py
X_train_test, y_train_test, X_val, y_val = None, None, None, None

# Stratified cross-validator
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Models
model_e = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model_e.classifier[1] = nn.Linear(model_e.classifier[1].in_features, 3, bias=True)

for p in model_e.parameters():
    p.requires_grad = True

model_r = resnet18(weights=ResNet18_Weights.DEFAULT)
model_r.fc = nn.Linear(model_r.fc.in_features, 3, bias=True)

for p in model_r.parameters():
    p.requires_grad = True

models = [
    ('EfficientNet_B0', model_e),
    ('ResNet_18', model_r),
]

# custom optimizer to encapsulate Adam
def make_lookahead(parameters, optimizer_cls, lr, weight_decay, **kwargs):
    optimizer = optimizer_cls(parameters, **kwargs)
    return Lookahead(optimizer=optimizer, lr=lr, weight_decay=weight_decay)

best_models = []
for name, model in models:
    net = NeuralNetClassifier(
        model,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        batch_size=64,
        optimizer=make_lookahead,
        optimizer__optimizer_cls=torch.optim.Adam,
        iterator_train__shuffle=True,
        verbose=0
    )

    params = {
        'max_epochs': [3, 6, 9, 12, 15],
        'optimizer__weight_decay': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        'optimizer__lr': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    }

    grid_search = GridSearchCV(net, params, refit=False, cv=skf, scoring='roc_auc_ovr', verbose=2)

    print("--> Starting Grid Search:", name)
    grid_search.fit(X_train_test, y_train_test)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    net.set_params(**best_params)
    best_models.append((name, net))

    net.fit(X_train_test, y_train_test)
    probabilities = net.predict_proba(X_val)
    predictions = net.predict(X_val)

    acc = accuracy_score(y_val, predictions)
    b_acc = balanced_accuracy_score(y_val, predictions)
    prec = precision_score(y_val, predictions, average="weighted")
    rec = recall_score(y_val, predictions, average="weighted")
    roc = roc_auc_score(y_val, probabilities, multi_class="ovr")
    cm =  confusion_matrix(y_val, predictions)

    print("--> Finishing Grid Search:", name)
    print("Best score: {:.3f}, best params: {}".format(best_score, best_params))
    print("Accuracy:", acc)
    print("Balanced Accuracy:", b_acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("ROC AUC:", roc)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                 display_labels=["Normal/Mid", "Moderate", "Severe"])
    disp.plot()
    plt.title(f'Confusion Matrix ({name})')
    plt.show()
    plt.close()

    train_loss = net.history[:, 'train_loss']
    valid_loss = net.history[:, 'valid_loss']

    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(valid_loss, label='Test Loss', color='orange')

    plt.title(f'Loss ({name})')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()
    plt.close()

# Ensemble

clf = StackingClassifier(
    estimators=best_models,
    final_estimator=RandomForestClassifier(class_weight='balanced', random_state=42),
    cv=skf,
    verbose=2
)
clf.fit(X_train_test, y_train_test)

probabilities = clf.predict_proba(X_val)
predictions = clf.predict(X_val)

acc = accuracy_score(y_val, predictions)
b_acc = balanced_accuracy_score(y_val, predictions)
prec = precision_score(y_val, predictions, average="weighted")
rec = recall_score(y_val, predictions, average="weighted")
roc = roc_auc_score(y_val, probabilities, multi_class="ovr")
cm =  confusion_matrix(y_val, predictions)

print("Accuracy:", acc)
print("Balanced Accuracy:", b_acc)
print("Precision:", prec)
print("Recall:", rec)
print("ROC AUC:", roc)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=["Normal/Mid", "Moderate", "Severe"])
disp.plot()
plt.title("Confusion Matrix (Stacking Ensemble)")
plt.show()
plt.close()

num_models = len(best_models)
fig, axes = plt.subplots(num_models, 1, figsize=(10, 5 * num_models), sharex=True)

for (name, model), ax in zip(best_models, axes):
    train_loss = model.history[:, 'train_loss']
    valid_loss = model.history[:, 'valid_loss']

    ax.plot(train_loss, label='Train Loss', color='blue')
    ax.plot(valid_loss, label='Test Loss', color='orange')

    ax.set_title(f'Losses ({name})')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

plt.xlabel('Epochs')
plt.tight_layout()
plt.show()
plt.close()