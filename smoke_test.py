"""Exercise the complete notebook using synthetic records only."""
from pathlib import Path
import contextlib
import io
import json
import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_fixture(n, seed, test=False):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    frame = pd.DataFrame({
        'Patient ID' if test else 'patient_id': np.arange(n),
        'age': rng.integers(20, 80, n),
        'bmi': rng.uniform(18, 36, n),
        'indicator_a': x,
        'indicator_b': rng.uniform(0, 1, n),
        'family_history_diabetes': rng.integers(0, 2, n),
        'category': rng.choice(['group-a', 'group-b'], n),
        'diagnosed_diabetes': (x > 0.5).astype(int),
        'diabetes_stage': pd.cut(x, [0, 1/3, 2/3, 1], labels=['stage-a', 'stage-b', 'stage-c']).astype(str),
        'diabetes_risk_score': 20 + 40 * x + rng.normal(0, 1, n),
    })
    if test:
        frame.loc[0, 'category'] = 'previously-unseen'
    return frame


def main():
    root = Path(__file__).resolve().parent
    notebook = json.loads((root / 'main.ipynb').read_text(encoding='utf-8'))
    original_data_dir = os.environ.get('DIABETES_DATA_DIR')
    namespace = {'__name__': '__main__'}
    executed = 0
    try:
        with tempfile.TemporaryDirectory(prefix='synthetic-diabetes-') as tmp:
            make_fixture(150, 42).to_csv(Path(tmp) / 'Diabetes.csv', index=False)
            make_fixture(45, 53, test=True).to_csv(Path(tmp) / 'Diabetes_test.csv', index=False)
            os.environ['DIABETES_DATA_DIR'] = tmp
            plt.show = lambda *args, **kwargs: plt.close('all')
            with contextlib.redirect_stdout(io.StringIO()):
                for i, cell in enumerate(notebook['cells']):
                    if cell['cell_type'] != 'code':
                        continue
                    code = ''.join(cell['source'])
                    code = '\n'.join(line for line in code.splitlines() if not line.lstrip().startswith('%'))
                    exec(compile(code, f'notebook-cell-{i}', 'exec'), namespace)
                    executed += 1
            assert executed == 9
            excluded = {'patient_id', 'Patient ID', 'diagnosed_diabetes', 'diabetes_stage', 'diabetes_risk_score'}
            assert excluded.isdisjoint(namespace['feature_cols'])
            assert len(namespace['results_binary']) == 3
            assert len(namespace['results_multi']) == 3
            assert len(namespace['results_regression']) == 3
            assert all(np.isfinite(v) for v in namespace['results_binary'].values())
            assert all(np.isfinite(v) for v in namespace['results_multi'].values())
            assert all(np.isfinite(v['rmse']) and np.isfinite(v['r2']) for v in namespace['results_regression'].values())
            assert set(namespace['model'].named_steps) == {'preprocess', 'pca', 'model'}
    finally:
        if original_data_dir is None:
            os.environ.pop('DIABETES_DATA_DIR', None)
        else:
            os.environ['DIABETES_DATA_DIR'] = original_data_dir
        plt.close('all')
    print('PASS: 9 notebook code cells; 9 models; identifiers/targets excluded; unseen category handled.')
    print('Synthetic fixtures only. No real-data or clinical performance claim.')


if __name__ == '__main__':
    main()
