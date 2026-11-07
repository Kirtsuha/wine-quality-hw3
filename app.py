import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent
st.set_page_config(page_title='Подборка вин', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def batch_predict(frame, model, columns, threshold):
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    features = frame[columns].apply(pd.to_numeric, errors='raise')
    if features.empty:
        raise ValueError('В файле нет строк с винами.')
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError('Признаки должны содержать конечные числа без пропусков.')
    if not features['is_red'].isin([0, 1]).all():
        raise ValueError('is_red должен быть равен 0 или 1.')
    p = model.predict_proba(features)[:, 1]
    result = frame.copy()
    result['probability'] = p
    result['prediction'] = (p >= threshold).astype(int)
    return result


model = load_model()
test = load_test()
X_test = test.drop(columns='target')
y_test = test['target']
p_test = test_proba(model, X_test)
config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))

st.title('Подборка хороших вин')
st.write('Хорошее вино — вино с экспертной оценкой не ниже 7.')
threshold = st.sidebar.slider('Порог', 0.0, 1.0, float(config['threshold']), 0.01)
st.sidebar.caption(f'Текущее значение: {threshold:.3f}')
one, quality, batch, weights_tab = st.tabs([
    'Одно вино', 'Модельные метрики', 'CSV с винами', 'Что выучила модель'])

with one:
    values = {}
    fields = st.columns(3)
    for i, name in enumerate(X_test.columns):
        with fields[i % 3]:
            if name == 'is_red':
                values[name] = int(st.checkbox('Красное вино'))
            else:
                values[name] = st.number_input(
                    name, min_value=float(X_test[name].min()),
                    max_value=float(X_test[name].max()),
                    value=float(X_test[name].median()), format='%.5f')
    sample = pd.DataFrame([values], columns=X_test.columns)
    probability = model.predict_proba(sample)[0, 1]
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    st.write('Включить в подборку' if probability >= threshold else 'Не включать в подборку')

with quality:
    labels = p_test >= threshold
    a, b, c = st.columns(3)
    a.metric('Precision', f'{metrics.precision_score(y_test, labels, zero_division=0):.3f}')
    b.metric('Recall', f'{metrics.recall_score(y_test, labels, zero_division=0):.3f}')
    c.metric('F1', f'{metrics.f1_score(y_test, labels, zero_division=0):.3f}')
    st.caption('Метрики на отложенной тестовой выборке. Порог для сдачи выбирался на валидации.')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cm = metrics.confusion_matrix(y_test, labels, labels=[0, 1])
    metrics.ConfusionMatrixDisplay(cm).plot(ax=axes[0], colorbar=False)
    fpr, tpr, _ = metrics.roc_curve(y_test, p_test)
    pr, re, _ = metrics.precision_recall_curve(y_test, p_test)
    tn, fp, fn, tp = cm.ravel()
    current_recall = tp / (tp + fn) if tp + fn else 0
    current_precision = tp / (tp + fp) if tp + fp else 0
    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].scatter(fp / (fp + tn), current_recall, color='red')
    axes[1].set(xlabel='FPR', ylabel='TPR', title='ROC')
    axes[2].plot(re, pr)
    axes[2].axhline(y_test.mean(), linestyle='--', color='gray')
    axes[2].scatter(current_recall, current_precision, color='red')
    axes[2].set(xlabel='Recall', ylabel='Precision', title='PR')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('При повышении порога в подборку попадает меньше вин. Обычно точность '
             'подборки растёт, но часть хороших вин теряется. При снижении порога '
             'подборка шире, но риск рекомендовать посредственное вино выше.')

with batch:
    st.write('Загрузите CSV с 12 признаками. Порог общий для всех вкладок.')
    st.caption('Нужные столбцы: ' + ', '.join(X_test.columns))
    uploaded = st.file_uploader('CSV с признаками вин', type=['csv'])
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
            result = batch_predict(frame, model, list(X_test.columns), threshold)
            st.dataframe(result)
            st.download_button('Скачать предсказания', result.to_csv(index=False).encode('utf-8'),
                               file_name='predictions.csv', mime='text/csv')
        except (ValueError, TypeError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(str(error))

with weights_tab:
    weights = pd.Series(model.steps[-1][1].coef_[0], index=X_test.columns).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    weights.plot.barh(ax=ax)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Коэффициент после StandardScaler')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('Сильнее всего за хорошее вино:', ', '.join(weights.nlargest(3).index))
    st.write('Сильнее всего против:', ', '.join(weights.nsmallest(3).index))
    st.caption('Коэффициенты описывают связи в этой модели и не доказывают причинность.')
