"""Початковий EDA для TUKE OZ: дані без цільових міток."""
from pathlib import Path
import argparse
import hashlib
import json
import platform

ROOT = Path(__file__).resolve().parents[1]


def analyze(data_path=None, output_dir=None):
    data_path = Path(data_path) if data_path is not None else ROOT / 'data/data.csv'
    output_dir = Path(output_dir) if output_dir is not None else ROOT / 'reports/eda'
    if not data_path.is_file():
        raise FileNotFoundError(
            f'CSV не знайдено: {data_path.resolve()}. '
            'Поклади файл Kaggle data.csv у папку data/ або передай --data.'
        )
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError('Встанови залежності: python -m pip install -r requirements.txt') from exc

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError('CSV не містить рядків для аналізу.')
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        'non_null': df.notna().sum(),
        'missing': df.isna().sum(),
        'missing_percent': df.isna().mean().mul(100),
        'unique_non_null': df.nunique(dropna=True),
    })
    profile.index.name = 'column'
    profile.to_csv(output_dir / 'column_profile.csv')
    df.describe(include='all').T.to_csv(output_dir / 'descriptive_statistics.csv')
    df.head(10).to_csv(output_dir / 'preview.csv', index=False)
    numeric = df.select_dtypes(include='number')
    if not numeric.empty:
        numeric.describe().T.to_csv(output_dir / 'numeric_statistics.csv')

    # ID зберігаємо у сирих даних; окремо перевіряємо дублікати без нього.
    # Інші можливі ідентифікатори потрібно оцінити після EDA вручну.
    id_columns = [c for c in df.columns if c.lower() == 'id']
    features = df.drop(columns=id_columns)
    digest = hashlib.sha256()
    with data_path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    summary = {
        'source': str(data_path.resolve()),
        'sha256': digest.hexdigest(),
        'task': 'unsupervised_clustering',
        'target_column': None,
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'missing_cells': int(df.isna().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'id_columns_excluded_from_feature_duplicate_check': id_columns,
        'duplicate_feature_rows': int(features.duplicated().sum()) if len(features.columns) else None,
        'constant_columns_including_missing': df.columns[df.nunique(dropna=False) <= 1].tolist(),
        'python_version': platform.python_version(),
        'pandas_version': pd.__version__,
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Звіти: {output_dir.resolve()}')
    return df, profile, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/data.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/eda')
    args = parser.parse_args()
    try:
        analyze(args.data, args.output)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
