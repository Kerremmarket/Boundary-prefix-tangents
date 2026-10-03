"""Reconcile the 30 transfers without modifying manuscript or frozen evidence.

Run with Python 3; standard library only. Read the stored decimal strings directly.
Percentages keep the selected lattice and its predictions fixed in both scenarios:
  loss:        100 * abs(Lq - L_selected) / abs(L_selected)
  interaction: 100 * abs(Jq - I_selected) / abs(I_selected)
The selected quadrature substitutes the sole warning refinement for its fine row.
This is a records reconciliation, not a rerun of the underlying evaluators.
"""
from collections import Counter
import csv
from decimal import Decimal, getcontext
import hashlib
import json
from pathlib import Path

getcontext().prec = 50
D = Decimal
ROOT = Path(__file__).resolve().parents[1]
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir', type=Path, default=ROOT / 'research/prefix_comparator_revision')
parser.add_argument('--output-dir', type=Path, default=ROOT / 'research/prefix_comparator_revision/results/quadrature-reconciliation')
args = parser.parse_args()
DATA = args.data_dir.resolve()
OUT = args.output_dir.resolve()
KEYS = ['date', 'contract_id', 'reference_model', 'transferred_from_model']


def read(relative):
    with (DATA / relative).open(newline='') as stream:
        return list(csv.DictReader(stream))


def key(row):
    return tuple(row[k] for k in KEYS)


def subset(rows, level, actual=False):
    kept = [r for r in rows if r['level'] == level
            and (not actual or D(r['eta']) == 1)]
    result = {key(r): r for r in kept}
    assert len(result) == len(kept), 'Duplicate cases'
    return result


def number(row, column):
    return D(row[column])


def gain(evaluation, predictions):
    loss = number(evaluation, 'finite_loss')
    return (abs(loss - number(predictions, 'additive_prediction'))
            - abs(loss - number(predictions, 'full_prefix_prediction')))


def write_csv(filename, rows):
    with (OUT / filename).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


raw = read('results/transfer_lattice_levels.csv')
qraw = read('results/transfer_quadrature_actual.csv')
stored = {key(r): r for r in read('tables/transfer_actual_all.csv')}
p = subset(raw, 'primary', actual=True)
f = subset(raw, 'fine', actual=True)
w = subset(raw, 'warning_r1', actual=True)
s = {**f, **w}
qf = subset(qraw, 'fine_quadrature')
qw = subset(qraw, 'warning_r1_quadrature')
qs = {**qf, **qw}
assert len(s) == 30 and len(w) == len(qw) == 1
assert set(p) == set(f) == set(s) == set(qf) == set(qs) == set(stored)
assert set(w) == set(qw)
assert next(iter(qw)) == ('2006-03-31', 'c10_g30_n10',
                         'annual_lognormal_mixture', 'black_scholes_atm')
assert all(int(r['intervals_per_local_cap']) == 8192
           and int(r['nodes_per_component']) == 256 for r in qf.values())
assert all(int(r['intervals_per_local_cap']) == 16384
           and int(r['nodes_per_component']) == 512 for r in qw.values())

rows, changes = [], []
for k in sorted(s):
    row = dict(zip(KEYS, k))
    row.update(eta='1', selected_lattice_level=s[k]['level'])
    for name in ['finite_loss', 'finite_interaction', 'additive_prediction',
                 'full_prefix_prediction', 'predicted_interaction']:
        row['selected_lattice_' + name] = number(s[k], name)
    assert number(s[k], 'finite_loss') != 0
    assert number(s[k], 'predicted_interaction') != 0
    for label, table in [('fine_quadrature', qf), ('refined_quadrature', qw),
                         ('selected_quadrature', qs)]:
        for name in ['finite_loss', 'finite_interaction', 'finite_singleton_sum']:
            row[label + '_' + name] = number(table[k], name) if k in table else ''
    for label, table in [('fine', qf), ('selected', qs)]:
        row[label + '_loss_percent'] = (
            100 * abs(number(table[k], 'finite_loss') - number(s[k], 'finite_loss'))
            / abs(number(s[k], 'finite_loss')))
        row[label + '_interaction_percent'] = (
            100 * abs(number(table[k], 'finite_interaction')
                      - number(s[k], 'predicted_interaction'))
            / abs(number(s[k], 'predicted_interaction')))
        row[label + '_gain_fixed_selected_predictions'] = gain(table[k], s[k])
    # Published fine-only Gain uses the fine lattice's own predictions.
    row['fine_gain_with_fine_predictions'] = gain(qf[k], f[k])
    row['selected_lattice_gain'] = gain(s[k], s[k])
    gp, gf, gs = gain(p[k], p[k]), gain(f[k], f[k]), gain(s[k], s[k])
    gqf, gqs = gain(qf[k], f[k]), gain(qs[k], s[k])
    floor = max(D('1e-11'), 10 * max(abs(gf-gp), abs(gs-gf), abs(gs-gqs),
                                    abs(gf-gqf), abs(gqs-gqf)))
    status = ('prefix_closer_resolved' if gs > 0 else 'additive_closer_resolved')
    if abs(gs) <= floor:
        status = 'unresolved'
    assert status == stored[k]['comparison_status']
    assert abs(floor - number(stored[k], 'gain_resolution_floor')) <= (
        D('1e-17') + D('3e-12') * abs(floor))
    for name in ['finite_loss', 'finite_interaction', 'predicted_interaction']:
        assert abs(number(s[k], name) - number(stored[k], name)) < D('1e-17')
    for name in ['finite_loss', 'finite_interaction']:
        assert abs(number(qs[k], name)
                   - number(stored[k], name + '_quadrature')) < D('1e-17')
    row['frozen_gain_resolution_floor_recomputed'] = floor
    row['frozen_comparison_status_recomputed'] = status
    rows.append(row)
    if k in qw:
        for name in qf[k]:
            if qf[k][name] != qw[k][name]:
                changes.append({**dict(zip(KEYS, k)), 'input': name,
                                'fine_value': qf[k][name], 'refined_value': qw[k][name]})

central, statistics = [], {}
for metric in ['loss', 'interaction']:
    for scenario in ['fine', 'selected']:
        col = scenario + '_' + metric + '_percent'
        ordered = sorted(rows, key=lambda row: (row[col], key(row)))
        stats = {'min_percent': ordered[0][col],
                 'median_percent': (ordered[14][col] + ordered[15][col]) / 2,
                 'max_percent': ordered[-1][col]}
        statistics[col] = stats
        for pos in [14, 15]:
            r = ordered[pos]
            central.append({'metric': metric, 'scenario': scenario, 'rank': pos + 1,
                            **{k: r[k] for k in KEYS}, 'percent': r[col],
                            'median_percent': stats['median_percent']})

count_columns = ['selected_lattice_gain', 'fine_gain_with_fine_predictions',
                 'fine_gain_fixed_selected_predictions',
                 'selected_gain_fixed_selected_predictions']
counts = {name: {'positive': sum(r[name] > 0 for r in rows),
                 'zero': sum(r[name] == 0 for r in rows),
                 'negative': sum(r[name] < 0 for r in rows)} for name in count_columns}
assert sum(r['fine_loss_percent'] != r['selected_loss_percent'] for r in rows) == 1
assert sum(r['fine_interaction_percent'] != r['selected_interaction_percent'] for r in rows) == 1
summary = {
    'scope': 'Reconciliation from frozen decimal records; no evaluator rerun.',
    'arithmetic': '50-digit Decimal arithmetic; no rounding before median.',
    'loss_formula': '100*abs(Lq-L_selected_lattice)/abs(L_selected_lattice)',
    'interaction_formula': '100*abs(Jq-I_selected_lattice)/abs(I_selected_lattice)',
    'fixed': ['30 actual-transfer cases (eta=1)', 'selected-lattice loss comparator',
              'selected-lattice predicted interaction', 'percentage denominators',
              'median convention: mean of ordered positions 15 and 16'],
    'changed_percentage_cases': 1,
    'statistics': statistics,
    'gain_counts': counts,
    'gain_sign_changes_fixed_predictions': sum(
        (r['fine_gain_fixed_selected_predictions'] > 0)
        != (r['selected_gain_fixed_selected_predictions'] > 0) for r in rows),
    'frozen_classifications_recomputed': dict(Counter(
        r['frozen_comparison_status_recomputed'] for r in rows)),
    'classification_note': 'The original resolution rule retains the entire ladder, '
        'including both quadrature levels. Re-labelling the discrepancy summary does '
        'not change that rule. No counterfactual rule deleting the refinement is asserted.',
    'source_sha256': {str(path.relative_to(DATA)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in [DATA/'results/transfer_lattice_levels.csv',
                                   DATA/'results/transfer_quadrature_actual.csv',
                                   DATA/'tables/transfer_actual_all.csv']},
}
OUT.mkdir(parents=True, exist_ok=True)
write_csv('reconciliation-30-rows.csv', rows)
write_csv('central-observations.csv', central)
write_csv('changed-quadrature-inputs.csv', changes)
(OUT/'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
print(json.dumps(summary, indent=2, default=str))
print('\nCENTRAL OBSERVATIONS')
for item in central:
    print(item)
