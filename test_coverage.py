import sys, json
sys.path.insert(0, '.')
from inference.run import run_culling_stage

result = run_culling_stage(
    input_dir        = r'C:\Users\erikg\Downloads\Jay Ma Photography Training\202602 NW CRC Raws',
    output_dir       = r'C:\Users\erikg\OneDrive\Desktop\CodePlayground\badminton-photo-editor\test_coverage_output',
    selection_target = 0.10,
    player_coverage  = True,
)

print()
print('=== RESULTS ===')
print('Total photos    :', result['total'])
print('Passed (scoring):', result['passed'] - result.get('coverage_stats', {}).get('n_promoted', 0))
print('Promoted by cov :', result.get('coverage_stats', {}).get('n_promoted', 0))
print('Final passed    :', result['passed'])
print('Culled          :', result['culled'])
cov = result.get('coverage_stats', {})
print('Clusters found  :', cov.get('n_clusters', '?'))
print('Full coverage   :', json.dumps(cov, indent=2))
