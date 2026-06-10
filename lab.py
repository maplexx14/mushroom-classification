# Импорты библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

import warnings
warnings.filterwarnings('ignore')

# random_state для того чтобы резы обучения были воспроизводимы при каждом запуске
RANDOM_STATE = 42


# Загрузка данных

# Датасет: Mushroom Classification (https://www.kaggle.com/datasets/uciml/mushroom-classification)
# 8124 гриба, 22 признака

df = pd.read_csv('mushrooms.csv')

print("=" * 55)
print("БЛОК 1: Загрузка данных")
print("=" * 55)
print(f"Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")
print("\nПервые 5 строк:")
print(df.head())
print("\nНазвания столбцов:")
print(list(df.columns))


# 2) EDA

print("\n" + "=" * 55)
print("БЛОК 2: EDA")
print("=" * 55)

print("\nТипы данных:")
print(df.dtypes.value_counts())

# Пропуски
print("\nПропущенные значения:", df.isnull().sum().sum())

# Распределение классов
print("\nРаспределение классов (class):")
print(df['class'].value_counts())
print(df['class'].value_counts(normalize=True).round(3))

# График 1: Распределение классов 
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle('EDA: Распределение классов', fontsize=13, fontweight='bold')

counts = df['class'].value_counts()
labels_pie = ['Съедобный (e)', 'Ядовитый (p)']
axes[0].pie(counts, labels=labels_pie, autopct='%1.1f%%',
            colors=['#27ae60', '#c0392b'], startangle=90)
axes[0].set_title('Доля классов')

axes[1].bar(labels_pie, counts.values,
            color=['#27ae60', '#c0392b'], edgecolor='black')
axes[1].set_title('Количество грибов по классам')
axes[1].set_ylabel('Количество')
for i, v in enumerate(counts.values):
    axes[1].text(i, v + 30, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('classes.png', dpi=150, bbox_inches='tight')
plt.show()

# График 2: 5 самых важных признаков по классам
top_features = ['odor', 'gill-color', 'spore-print-color', 'gill-size', 'ring-type']

fig, axes = plt.subplots(1, len(top_features), figsize=(18, 5))
fig.suptitle('EDA: Ключевые признаки по классам', fontsize=13, fontweight='bold')

for ax, feat in zip(axes, top_features):
    ct = pd.crosstab(df[feat], df['class'])
    ct.plot(kind='bar', ax=ax, color=['#27ae60', '#c0392b'],
            edgecolor='black', legend=False)
    ax.set_title(feat)
    ax.set_xlabel('')
    ax.set_ylabel('Кол-во')
    ax.tick_params(axis='x', rotation=0)

axes[0].legend(['Съедобный', 'Ядовитый'], fontsize=8)
plt.tight_layout()
plt.savefig('features.png', dpi=150, bbox_inches='tight')
plt.show()

# График 3: Признак 'odor' самый важный
# almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
plt.figure(figsize=(9, 4))
odor_map = {'n': 'нет', 'a': 'анис', 'l': 'миндаль', 'f': 'вонь',
            'c': 'хлор', 'y': 'рыба', 's': 'острый', 'p': 'перец', 'm': 'мускус'}
df_odor = df.copy()
df_odor['odor_name'] = df_odor['odor'].map(odor_map).fillna(df_odor['odor'])

ct = pd.crosstab(df_odor['odor_name'], df_odor['class'])
ct.plot(kind='bar', color=['#27ae60', '#c0392b'], edgecolor='black')
plt.title('EDA: Запах гриба и класс', fontsize=12, fontweight='bold')
plt.xlabel('Запах')
plt.ylabel('Количество грибов')
plt.legend(['Съедобный', 'Ядовитый'])
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('odor.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nВыводы EDA:")
print("Классы почти сбалансированы: ~52% съедобных, ~48% ядовитых")
print("Пропусков нет")
print("Признак 'odor' очень хорошо разделяет классы:")
print("острый/вонь/хлор/рыба = почти всегда ядовитый")
print("анис/миндаль/нет = съедобный")

print("\nПлан действий:")
print("Пропусков нет - дополнительная обработка не требуется")
print("Выбросов нет (признаки категориальные)")
print("Классы сбалансированы — методы балансировки не применяются")
print("Категориальные признаки будут закодированы")
print("Данные будут разделены на обучающую и тестовую выборки")
print("Будут обучены модели классификации и проведена оценка качества")


# 3) Предобработка данных

print("\n" + "=" * 55)
print("БЛОК 3: Предобработка")
print("=" * 55)

df_proc = df.copy()

# Кодируем целевую переменную: e = 0 (съедобный), p = 1 (ядовитый)
df_proc['class'] = df_proc['class'].map({'e': 0, 'p': 1})
print("Целевая переменная: e=0 (съедобный), p=1 (ядовитый)")

# Все остальные признаки - категориальные, применяем Label Encoding(меняем каждую категорию цифрой)
# Подходит для деревьев, которые не предполагают упорядоченности
le = LabelEncoder()
feature_cols = [c for c in df_proc.columns if c != 'class']

for col in feature_cols:
    df_proc[col] = le.fit_transform(df_proc[col])

print(f"\nLabel Encoding применён к {len(feature_cols)} признакам")
print("Пример после кодирования:")
print(df_proc.head(3))

# Разделение на признаки и цель
X = df_proc[feature_cols]
y = df_proc['class']

# train_test_split: 80% обучение, 20% тест
# stratify=y - сохраняем пропорцию классов в обеих выборках
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\nОбучающая выборка:{X_train.shape[0]} объектов")
print(f"Тестовая выборка: {X_test.shape[0]} объектов")

# Нормализация не нужна - деревья не зависят от масштаба признаков
print("Нормализация не применяется, так как используются алгоритмы на деревьях,")
print("которые принимают решения на основе пороговых сравнений и не зависят от масштаба признаков")


# 4) Обучение моделей

print("\n" + "=" * 55)
print("БЛОК 4: Обучение моделей")
print("=" * 55)

# Baseline 
# Всегда предсказывает мажоритарный класс
majority = y_train.mode()[0]
y_baseline = np.full(len(y_test), majority)
baseline_acc = accuracy_score(y_test, y_baseline)
print(f"\nBaseline (всегда класс {majority}): Accuracy = {baseline_acc:.3f}")

# Дерево решений
# Почему: дерево хорошо работает с категориальными данными, результат легко интерпретировать
# ядовитость гриба по цепочке условий.
# Тестируем разные значения max_depth
print("\nПодбор max_depth для Дерева решений:")
for depth in [2, 3, 5, 10, None]:
    dt_tmp = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    score = cross_val_score(dt_tmp, X_train, y_train, cv=5, scoring='f1').mean()
    print(f"  max_depth={str(depth):5s} --> CV F1 = {score:.4f}")

# max_depth=5 даёт хороший результат и не переобучается
dt = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
print("\nДерево решений (max_depth=5) обучено")


# Случайный лес 
# Ансамбль из многих деревьев устойчивее к случайным ошибкам.
# n_estimators=100: 100 деревьев
# max_depth=10: не даём деревьям расти слишком глубоко (>30 уже перебор, а <10 слишком простое дерево)
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)
print("Случайный лес (100 деревьев) обучен")

# Логистическая регрессия
# На категориальных данных обычно уступает деревьям.
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train, y_train)
print("Логистическая регрессия обучена")


# 5) Оценка качества

print("\n" + "=" * 55)
print("БЛОК 5: Оценка качества")
print("=" * 55)

def get_metrics(y_true, y_pred, name):
    m = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
    }
    print(f"\n {name}")
    print(f" {'─'*35}")
    for k, v in m.items():
        print(f" {k:12s}: {v:.4f}")
    return m

m_base = get_metrics(y_test, y_baseline, "Baseline")
m_dt = get_metrics(y_test, dt.predict(X_test), "Дерево решений")
m_rf = get_metrics(y_test, rf.predict(X_test), "Случайный лес")
m_lr = get_metrics(y_test, lr.predict(X_test), "Логистическая регрессия")

# Classification report
print("\nПодробный отчёт (Случайный лес):")
print(classification_report(
    y_test, rf.predict(X_test),
    target_names=['Съедобный', 'Ядовитый']
))

# График 4: Матрицы ошибок
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Матрицы ошибок', fontsize=13, fontweight='bold')

for ax, (name, y_pred) in zip(axes, [
    ("Дерево решений", dt.predict(X_test)),
    ("Случайный лес", rf.predict(X_test)),
    ("Логистическая регрессия", lr.predict(X_test)),
]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                xticklabels=['Съедобный', 'Ядовитый'],
                yticklabels=['Съедобный', 'Ядовитый'])
    ax.set_title(name)
    ax.set_xlabel('Предсказание')
    ax.set_ylabel('Реальность')

plt.tight_layout()
plt.savefig('matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# График 5: Сравнение метрик 
model_names = ['Baseline', 'Дерево решений', 'Случайный лес', 'Лог. регрессия']
all_metrics = [m_base, m_dt, m_rf, m_lr]
keys = ['Accuracy', 'Precision', 'Recall', 'F1']
comp = pd.DataFrame(all_metrics, index=model_names)[keys]

print("\nТаблица сравнения:")
print(comp.round(4).to_string())

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(keys))
w = 0.18
colors = ['#95a5a6', '#e67e22', '#27ae60', '#3498db']

for i, (name, row) in enumerate(comp.iterrows()):
    ax.bar(x + i * w, row.values, w, label=name,
           color=colors[i], edgecolor='black', linewidth=0.5)

ax.set_xticks(x + w * 1.5)
ax.set_xticklabels(keys, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Значение')
ax.set_title('Сравнение моделей по метрикам', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('metrics.png', dpi=150, bbox_inches='tight')
plt.show()

# График 6: Важность признаков в Random Forest
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
top15 = feat_imp.tail(15)  # 15 самых важных

plt.figure(figsize=(9, 6))
colors_bar = ['#e74c3c' if v > top15.quantile(0.66) else '#3498db' for v in top15]
top15.plot(kind='barh', color=colors_bar, edgecolor='black', linewidth=0.5)
plt.title('15 важных признаков (Random Forest)', fontsize=12, fontweight='bold')
plt.xlabel('Важность')
plt.tight_layout()
plt.savefig('importance.png', dpi=150, bbox_inches='tight')
plt.show()


# 6) Анализ ошибок

print("\n" + "=" * 55)
print("БЛОК 6: Анализ ошибок")
print("=" * 55)

y_pred_rf = rf.predict(X_test)
errors = np.where(y_pred_rf != y_test.values)[0]
print(f"\nОшибок Random Forest: {len(errors)} из {len(y_test)}")

if len(errors) > 0:

    print("\nПричины возможных ошибок:")
    print("Некоторые признаки могут иметь схожие комбинации у съедобных и ядовитых грибов")
    print("Модель может путаться в редких сочетаниях категорий,")
    print("которые редко встречались в обучающей выборке")
    print("Возможна ограниченная глубина деревьев (max_depth),")
    print("что не позволяет учесть все сложные зависимости")
    err_df = X_test.iloc[errors].copy()
    err_df['True'] = y_test.values[errors]
    err_df['Pred'] = y_pred_rf[errors]

    fn_count = ((err_df['True'] == 1) & (err_df['Pred'] == 0)).sum()
    fp_count = ((err_df['True'] == 0) & (err_df['Pred'] == 1)).sum()
    print(f"False Negative (ядовитый съедобный): {fn_count}")
    print(f"False Positive (съедобный ядовитый): {fp_count}")

else:
  print("\nОшибок нет, так как есть выраженный признак odor, который почти всегда показывает ядовитость")
  print("Не переобучение, потому что и на тестовых данных также 100% точность + модель ограничена по глубине")


# Итоги

print("\n" + "=" * 55)
print("Итоги")
print("=" * 55)
best = comp['F1'].idxmax()
print(f"Лучшая модель: {best}")
print(f"F1: {comp.loc[best, 'F1']:.4f}")
print(f"Baseline F1: {m_base['F1']:.4f}")


improvement = comp.loc[best, 'F1'] - m_base['F1']

print(f"Улучшение по F1 относительно baseline: +{improvement:.4f}")
print("Baseline предсказывает только мажоритарный класс и игнорирует признаки")
print("Модели машинного обучения используют информацию о признаках грибов," )
print("поэтому демонстрируют значительно более высокие значения Precision,")
print("Recall и F1-score")

print("\nВывод: деревья решений и случайный лес отлично справляются")
print("с категориальными данными без нормализации")
print("Признак 'odor'(запах) - самый показательный")