"""Дослідження ознак і числова baseline-підготовка для кластеризації."""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn

ROOT = Path(__file__).resolve().parents[1]


def prepare(root=ROOT):
    root = Path(root)
    source = root / 'data/data.csv'
    digest_before = hashlib.sha256(source.read_bytes()).hexdigest()
    df = pd.read_csv(source)
    if 'id' not in df or df['id'].isna().any() or not df['id'].is_unique:
        raise ValueError('Очікується унікальний непорожній id; перевір дані.')
    X = df.drop(columns='id')
    if X.empty or not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
        raise ValueError('Цей baseline очікує непорожню таблицю числових ознак.')
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError('Є пропуски або нескінченні значення; потрібне окреме рішення.')
    if (X.nunique() <= 1).any():
        raise ValueError('Є сталі ознаки; переглянь їх перед масштабуванням.')

    report = root / 'reports/preprocessing'
    processed = root / 'data/processed'
    report.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    integer = X.select_dtypes(include='integer').columns.tolist()
    continuous = [c for c in X if c not in integer]
    q1, q3 = X.quantile(.25), X.quantile(.75)
    iqr = q3 - q1
    outside = (X.lt(q1 - 1.5 * iqr) | X.gt(q3 + 1.5 * iqr))
    profile = pd.DataFrame({
        'dtype': X.dtypes.astype(str), 'unique': X.nunique(),
        'min': X.min(), 'max': X.max(), 'mean': X.mean(),
        'std_population': X.std(ddof=0), 'skew': X.skew(),
        'iqr': iqr, 'outside_1_5_iqr_count': outside.sum(),
        'outside_1_5_iqr_percent': outside.mean() * 100,
        'observed_type': ['integer_discrete_candidate' if c in integer else 'continuous_candidate' for c in X],
    })
    profile.index.name = 'feature'
    profile.to_csv(report / 'feature_profile.csv')

    for columns, filename, title in [
        (continuous, 'continuous_distributions.png', 'Continuous candidates — all rows'),
        (integer, 'discrete_distributions.png', 'Integer features — exact value counts'),
    ]:
        rows = (len(columns) + 3) // 4
        fig, axes = plt.subplots(rows, 4, figsize=(16, rows * 2.6), squeeze=False)
        for ax, col in zip(axes.flat, columns):
            if col in integer:
                counts = X[col].value_counts().sort_index()
                ax.bar(counts.index, counts.values, width=.85, color='#168b86')
            else:
                ax.hist(X[col], bins=60, color='#3479b7', edgecolor='none')
            ax.set_title(col)
            ax.set_ylabel('Rows')
            ax.grid(axis='y', alpha=.15)
        for ax in list(axes.flat)[len(columns):]:
            ax.set_visible(False)
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(report / filename, dpi=140)
        plt.close(fig)

    corr = X.corr(method='pearson')
    corr.to_csv(report / 'correlations_pearson.csv')
    pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(X.columns) for b in X.columns[i+1:]]
    pairs = pd.DataFrame(pairs, columns=['feature_a', 'feature_b', 'pearson_r'])
    pairs['absolute_r'] = pairs.pearson_r.abs()
    pairs = pairs.sort_values('absolute_r', ascending=False).reset_index(drop=True)
    pairs.to_csv(report / 'correlation_pairs.csv', index=False)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(X.columns)))
    ax.set_xticklabels(X.columns, rotation=90)
    ax.set_yticks(range(len(X.columns)))
    ax.set_yticklabels(X.columns)
    ax.set_title('Pearson correlation — id excluded')
    fig.colorbar(im, ax=ax, label='Pearson r', shrink=.8)
    fig.tight_layout()
    fig.savefig(report / 'correlations.png', dpi=140)
    plt.close(fig)

    # Трансдуктивний baseline: досліджуємо всі надані рядки без target.
    # Для оцінки на нових даних scaler слід fit лише на тренувальній частині.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    assert np.isfinite(scaled).all()
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-10)
    assert np.allclose(scaler.inverse_transform(scaled), X.to_numpy())
    np.savez_compressed(processed / 'numeric_standardized.npz',
                        X=scaled, ids=df['id'].to_numpy(), columns=np.asarray(X.columns, dtype=str))
    parameters = {'method': 'StandardScaler', 'feature_order': X.columns.tolist(),
                  'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                  'variance': scaler.var_.tolist(), 'n_samples_seen': int(scaler.n_samples_seen_)}
    (processed / 'scaler_parameters.json').write_text(json.dumps(parameters, indent=2) + '\n')
    pd.DataFrame({'mean_before': X.mean(), 'std_before': X.std(ddof=0),
                  'mean_after': scaled.mean(axis=0), 'std_after': scaled.std(axis=0)},
                 index=X.columns).to_csv(report / 'scaling_check.csv', index_label='feature')
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(range(len(X.columns)), X.std(ddof=0), 'o-', label='Before scaling')
    ax.plot(range(len(X.columns)), scaled.std(axis=0), 'o-', label='After scaling')
    ax.set_xticks(range(len(X.columns)))
    ax.set_xticklabels(X.columns, rotation=90)
    ax.set_ylabel('Population standard deviation')
    ax.set_title('Feature scales before and after StandardScaler')
    ax.legend()
    fig.tight_layout()
    fig.savefig(report / 'scaling.png', dpi=140)
    plt.close(fig)

    summary = {
        'source': str(source.resolve()), 'source_sha256': digest_before,
        'rows': len(X), 'features': len(X.columns), 'excluded_columns': ['id'],
        'integer_candidates': integer, 'continuous_candidates': continuous,
        'rows_outside_iqr_in_any_feature': int(outside.any(axis=1).sum()),
        'removed_rows': 0, 'max_absolute_off_diagonal_pearson': float(pairs.absolute_r.max()),
        'assumption': 'Integer features treated as numeric for this baseline; categorical semantics unconfirmed.',
        'fit_scope': 'All rows, transductive clustering; no held-out evaluation claimed.',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__},
    }
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest_before
    with np.load(processed / 'numeric_standardized.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['ids'], df['id'].to_numpy())
        assert np.array_equal(saved['columns'], X.columns.to_numpy())
        assert np.allclose(saved['X'], scaled)
    (report / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return profile, pairs, summary


if __name__ == '__main__':
    _, _, summary = prepare()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
