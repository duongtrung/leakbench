"""Re-run RQ5 validation after the channel-sampler fix and rewrite its stage JSON."""
import json
from leakbench.experiments.config import ExperimentConfig
from leakbench.experiments.runners import run_validation
cfg = ExperimentConfig(seed=20270214)
out = run_validation(cfg, n_per_class=3000)
# Keep the stage's own provenance block; only the note changes.
prov = json.load(open("outputs/scale_study/validation.json"))["provenance"]
prov["note"] = "paper-scale study; re-run after the channel-sampler fix"
payload = {"status": "real_computation_at_paper_scale", "result": out, "provenance": prov}
json.dump(payload, open("outputs/scale_study/validation.json", "w"), indent=2)
print("max |err| =", round(out["max_mi_abs_error"], 4))
for r in out["rows"]:
    print(f"p={r['p']:.2f}  MI analytic={r['analytic_mi_bits']:.3f} est={r['estimated_mi_bits']:.3f}"
          f"   risk analytic={r['analytic_bayes_risk']:.3f} est={r['estimated_bayes_risk']:.3f}")
