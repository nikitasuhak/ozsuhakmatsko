"""Reproducible transductive clustering baseline for TUKE OZ."""
from pathlib import Path
from time import perf_counter
import hashlib
import itertools
import json
import warnings
import platform

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             pairwise_distances)
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
KS = range(2, 11)


def make_model(family, k, seed=SEED, final=False):
    if family == 'KMeans':
        return KMeans(n_clusters=k, random_state=seed, n_init=10 if final else 5,
                      max_iter=300)
    return GaussianMixture(n_components=k, covariance_type='full',
                           random_state=seed, n_init=2 if final else 1,
                           max_iter=200, reg_covar=1e-6, tol=1e-3)


def fit(model, X):
    start = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(X)
    messages = [str(w.message) for w in caught]
    converged = bool(getattr(model, 'converged_', model.n_iter_ < model.max_iter))
    return perf_counter() - start, converged, messages


def scores(X, labels, sample_indices, distances):
    unique, counts = np.unique(labels, return_counts=True)
    sub = labels[sample_indices]
    valid = 1 < len(unique) < len(X)
    return {
        'silhouette_sample': float(silhouette_score(distances, sub, metric='precomputed'))
        if 1 < len(np.unique(sub)) < len(sub) else np.nan,
        'calinski_harabasz': float(calinski_harabasz_score(X, labels)) if valid else np.nan,
        'davies_bouldin': float(davies_bouldin_score(X, labels)) if valid else np.nan,
        'observed_clusters': len(unique), 'smallest_cluster': int(counts.min()),
        'largest_cluster': int(counts.max()),
    }


def run(root=ROOT):
    root = Path(root)
    report = root / 'reports/modeling'
    models = root / 'models'
    report.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    with np.load(root / 'data/processed/numeric_standardized.npz', allow_pickle=False) as a:
        X, ids, columns = a['X'], a['ids'], a['columns']
    raw_path = root / 'data/data.csv'
    source_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    prep = json.loads((root / 'reports/preprocessing/summary.json').read_text())
    if source_hash != prep['source_sha256']:
        raise ValueError('Raw CSV changed: rerun preprocessing first.')
    raw = pd.read_csv(raw_path)
    parameters = json.loads((root / 'data/processed/scaler_parameters.json').read_text())
    assert np.array_equal(ids, raw['id'].to_numpy())
    assert columns.tolist() == parameters['feature_order']
    expected = (raw[columns].to_numpy() - np.asarray(parameters['mean'])) / np.asarray(parameters['scale'])
    assert X.shape == expected.shape and np.allclose(X, expected)
    assert np.isfinite(X).all() and 'id' not in columns
    rng = np.random.default_rng(SEED)
    fit_idx = np.sort(rng.choice(len(X), min(20000, len(X)), replace=False))
    train = X[fit_idx]
    eval_idx = np.sort(rng.choice(len(train), min(2000, len(train)), replace=False))
    distances = pairwise_distances(train[eval_idx])
    np.fill_diagonal(distances, 0)
    np.savez_compressed(report / 'sampling_indices.npz', fit_indices=fit_idx,
                        silhouette_indices_in_fit=eval_idx)
    rows = []
    for family in ['KMeans', 'GaussianMixture']:
        for k in KS:
            model = make_model(family, k)
            elapsed, converged, messages = fit(model, train)
            labels = model.predict(train)
            row = {'family': family, 'k': k, 'fit_rows': len(train), 'seed': SEED,
                   'fit_seconds': elapsed, 'converged': converged,
                   'iterations': int(model.n_iter_), 'warnings': '; '.join(messages)}
            row.update(scores(train, labels, eval_idx, distances))
            row.update(inertia=float(model.inertia_) if family == 'KMeans' else np.nan,
                       bic=float(model.bic(train)) if family == 'GaussianMixture' else np.nan,
                       aic=float(model.aic(train)) if family == 'GaussianMixture' else np.nan)
            rows.append(row)
            pd.DataFrame(rows).to_csv(report / 'search_metrics.csv', index=False)
            print(f'{family} k={k}: silhouette={row["silhouette_sample"]:.4f}, converged={converged}', flush=True)
    search = pd.DataFrame(rows)
    candidates = search[search.converged & (search.observed_clusters == search.k)]
    chosen = {}
    for family in ['KMeans', 'GaussianMixture']:
        group = candidates[candidates.family == family]
        if group.empty:
            raise RuntimeError(f'No converged candidates for {family}. Inspect search_metrics.csv.')
        criterion = 'silhouette_sample' if family == 'KMeans' else 'bic'
        group = group.dropna(subset=[criterion]).sort_values(criterion, ascending=family != 'KMeans')
        chosen[family] = int(group.iloc[0].k)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for family in chosen:
        g = search[search.family == family]
        axes[0].plot(g.k, g.silhouette_sample, 'o-', label=family)
    axes[0].set_title('Silhouette on same 2,000 rows (higher is better)')
    axes[0].legend()
    km = search[search.family == 'KMeans']
    gm = search[search.family == 'GaussianMixture']
    axes[1].plot(km.k, km.inertia, 'o-')
    axes[1].set_title('K-Means inertia (inspect the elbow)')
    axes[2].plot(gm.k, gm.bic, 'o-', label='BIC')
    axes[2].plot(gm.k, gm.aic, 'o-', label='AIC')
    axes[2].set_title('GMM information criteria (lower is better)')
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel('Number of clusters k')
        ax.set_xticks(list(KS))
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(report / 'model_selection.png', dpi=150)
    plt.close(fig)

    stability = []
    for family, k in chosen.items():
        predictions = {}
        for seed in [42, 43, 44]:
            model = make_model(family, k, seed)
            _, converged, messages = fit(model, train)
            predictions[seed] = (model.predict(train), converged)
        for a, b in itertools.combinations(predictions, 2):
            stability.append({'family': family, 'k': k, 'seed_a': a, 'seed_b': b,
                              'both_converged': predictions[a][1] and predictions[b][1],
                              'adjusted_rand_index': adjusted_rand_score(predictions[a][0], predictions[b][0])})
    pd.DataFrame(stability).to_csv(report / 'seed_stability.csv', index=False)

    full_eval = np.sort(rng.choice(len(X), min(2000, len(X)), replace=False))
    full_distances = pairwise_distances(X[full_eval])
    np.fill_diagonal(full_distances, 0)
    np.save(report / 'final_silhouette_indices.npy', full_eval)
    assignments = pd.DataFrame({'id': ids})
    final_rows = []
    fitted = {}
    for family, k in chosen.items():
        print(f'Final fit: {family}, k={k}, rows={len(X)}', flush=True)
        model = make_model(family, k, final=True)
        elapsed, converged, messages = fit(model, X)
        labels = model.predict(X)
        assert len(labels) == len(ids) and np.isfinite(labels).all()
        row = {'family': family, 'k': k, 'fit_rows': len(X), 'converged': converged,
               'fit_seconds': elapsed, 'warnings': '; '.join(messages)}
        row.update(scores(X, labels, full_eval, full_distances))
        final_rows.append(row)
        assignments[family] = labels
        if family == 'GaussianMixture':
            assignments['gmm_max_probability'] = model.predict_proba(X).max(axis=1)
        joblib.dump(model, models / f'{family}.joblib')
        reloaded = joblib.load(models / f'{family}.joblib')
        assert np.array_equal(reloaded.predict(X[:100]), labels[:100])
        fitted[family] = model
        counts = pd.Series(labels).value_counts().sort_index().rename_axis('cluster').reset_index(name='rows')
        counts['percent'] = counts.rows / len(X) * 100
        counts.to_csv(report / f'{family}_cluster_sizes.csv', index=False)
        pd.DataFrame(X, columns=columns).groupby(labels).mean().to_csv(report / f'{family}_standardized_profiles.csv', index_label='cluster')
        raw[columns].groupby(labels).mean().to_csv(report / f'{family}_raw_profiles.csv', index_label='cluster')
    assignments.to_csv(report / 'cluster_assignments.csv', index=False)
    final = pd.DataFrame(final_rows)
    final.to_csv(report / 'final_metrics.csv', index=False)

    plot_idx = np.sort(rng.choice(len(X), min(10000, len(X)), replace=False))
    pca = PCA(n_components=2, svd_solver='full').fit(X)
    points = pca.transform(X[plot_idx])
    np.save(report / 'pca_plot_indices.npy', plot_idx)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, family in zip(axes, chosen):
        labels = assignments[family].to_numpy()[plot_idx]
        ax.scatter(points[:, 0], points[:, 1], c=labels, cmap='tab10', s=3, alpha=.4, rasterized=True)
        ax.set_title(f'{family}, k={chosen[family]}')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    fig.suptitle('Same PCA projection; colors are arbitrary cluster IDs')
    fig.tight_layout()
    fig.savefig(report / 'pca_clusters.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, family in zip(axes, chosen):
        counts = assignments[family].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color='#248a91')
        ax.set_title(family)
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Rows')
    fig.tight_layout()
    fig.savefig(report / 'cluster_sizes.png', dpi=150)
    plt.close(fig)
    limitations = [
        'Transductive baseline; no held-out performance or ground-truth accuracy claimed.',
        'Scaler fitted on all rows in stage 02. Search uses a fixed subset, not an independent test set.',
        'Integer features treated as numeric; semantics remain unconfirmed.',
        'Seed stability compares initializations on the same sample, not resampling stability.',
        'PCA is a lossy 2D illustration, not evidence of true cluster separation.',
        'Different selection criteria per family; no universal winning model is declared.',
    ]
    if any(k in [min(KS), max(KS)] for k in chosen.values()):
        limitations.append('A selected k lies on a search boundary: extend the range before claiming an optimum.')
    if not all(final.converged):
        limitations.append('A final model did not converge: inspect final_metrics.csv and refit before use.')
    summary = {'source_sha256': source_hash, 'seed': SEED, 'search_rows': len(train),
               'total_rows': len(X), 'features': X.shape[1], 'k_values': list(KS),
               'selected_k': chosen, 'selection_rules': {'KMeans': 'maximum sample silhouette', 'GaussianMixture': 'minimum BIC'},
               'final_metrics': final.to_dict(orient='records'),
               'seed_stability': stability,
               'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
               'model_agreement_ari': adjusted_rand_score(assignments.KMeans, assignments.GaussianMixture),
               'limitations': limitations,
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__}}
    (report / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == source_hash
    return search, final, pd.DataFrame(stability), summary


def main(root=ROOT):
    with threadpool_limits(limits=2):
        return run(root)


if __name__ == '__main__':
    main()
