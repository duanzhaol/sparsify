### L-1: Continue trial_0001 from 15M to 30M tokens on the baseline K80/NUM_EXPERTS326/LATENTS_PER_EXPERT96 recipe; latest_exceed
- **Strategy:** Continue trial_0001 from 15M to 30M tokens on the baseline K80/NUM_EXPERTS326/LATENTS_PER_EXPERT96 recipe; latest_exceed objective improved and the scheduler retained continue_current for the same trial.
- **Outcome:** keep
- **Insight:** Continue trial_0001 from 15M to 30M tokens on the baseline K80/NUM_EXPERTS326/LATENTS_PER_EXPERT96 recipe; latest_exceed objective improved and the scheduler retained continue_current for the same trial.
- **Context:** goal=持续优化 Qwen3-4B layer17 q_proj SAE 部署 objective（latest_exceed 语义）; scope=research/deploy_objective_autoresearch/**,research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/**,checkpoints/qwen3-4B/deploy_objective_autoresearch/**,tests/research/test_deploy_objective_*.py; metric=objective = total_cost_ratio + latest_exceed_alpha_0.50; direction=lower
- **Iteration:** qwen3-4b-layer17-qproj-deploy-objective#1
- **Timestamp:** 2026-04-13T17:40:32Z

### L-2: trial_0001 second retained checkpoint on the baseline K80/NUM_EXPERTS326/LATENTS_PER_EXPERT96 recipe reduced the latest-
- **Strategy:** trial_0001 second retained checkpoint on the baseline K80/NUM_EXPERTS326/LATENTS_PER_EXPERT96 recipe reduced the latest-exceed deploy objective from 0.5752488687 to 0.5747441366 while staying on the same trial.
- **Outcome:** keep
- **Insight:** trial_0001 second retained checkpoint on the baseline K80/NUM_EXPERTS326/LATENTS_PER_EXPERT96 recipe reduced the latest-exceed deploy objective from 0.5752488687 to 0.5747441366 while staying on the same trial.
- **Context:** goal=持续优化 Qwen3-4B layer17 q_proj SAE 部署 objective（latest_exceed 语义）; scope=research/deploy_objective_autoresearch/**,research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/**,checkpoints/qwen3-4B/deploy_objective_autoresearch/**,tests/research/test_deploy_objective_*.py; metric=objective = total_cost_ratio + latest_exceed_alpha_0.50; direction=lower
- **Iteration:** qwen3-4b-layer17-qproj-deploy-objective#2
- **Timestamp:** 2026-04-13T17:45:42Z

### L-3: Continue trial_0001 from 30M to 45M tokens on the unchanged baseline recipe; latest_exceed objective improved again and
- **Strategy:** Continue trial_0001 from 30M to 45M tokens on the unchanged baseline recipe; latest_exceed objective improved again and the scheduler retained continue_current for a fourth 15M segment.
- **Outcome:** keep
- **Insight:** Continue trial_0001 from 30M to 45M tokens on the unchanged baseline recipe; latest_exceed objective improved again and the scheduler retained continue_current for a fourth 15M segment.
- **Context:** goal=持续优化 Qwen3-4B layer17 q_proj SAE 部署 objective（latest_exceed 语义）; scope=research/deploy_objective_autoresearch/**,research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/**,checkpoints/qwen3-4B/deploy_objective_autoresearch/**,tests/research/test_deploy_objective_*.py; metric=objective = total_cost_ratio + latest_exceed_alpha_0.50; direction=lower
- **Iteration:** qwen3-4b-layer17-qproj-deploy-objective#3
- **Timestamp:** 2026-04-13T17:48:40Z

### L-4: [PIVOT] Abandon the higher-cost K=80/88 continuation strategy after five consecutive non-keep outer iterations: K=88 red
- **Strategy:** [PIVOT] Abandon the higher-cost K=80/88 continuation strategy after five consecutive non-keep outer iterations: K=88 reduced latest_exceed but never offset its extra cost and regressed again at 60M. Switch to an under-baseline cost frontier, starting from a cheaper K=72 candidate with the same experts/latents to test whether comparable exceed can convert into a lower deploy objective.
- **Outcome:** pivot
- **Insight:** [PIVOT] Abandon the higher-cost K=80/88 continuation strategy after five consecutive non-keep outer iterations: K=88 reduced latest_exceed but never offset its extra cost and regressed again at 60M. Switch to an under-baseline cost frontier, starting from a cheaper K=72 candidate with the same experts/latents to test whether comparable exceed can convert into a lower deploy objective.
- **Context:** goal=持续优化 Qwen3-4B layer17 q_proj SAE 部署 objective（latest_exceed 语义）; scope=research/deploy_objective_autoresearch/**,research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/**,checkpoints/qwen3-4B/deploy_objective_autoresearch/**,tests/research/test_deploy_objective_*.py; metric=objective = total_cost_ratio + latest_exceed_alpha_0.50; direction=lower
- **Iteration:** qwen3-4b-layer17-qproj-deploy-objective#10
- **Timestamp:** 2026-04-13T18:33:47Z

### L-5: Continue pivoted trial_0003 from 30M to 45M tokens at K=72; the cheaper-than-baseline recipe reduced latest_exceed enoug
- **Strategy:** Continue pivoted trial_0003 from 30M to 45M tokens at K=72; the cheaper-than-baseline recipe reduced latest_exceed enough to reach a new outer best objective of 0.5686420077, beating the retained 0.5721993461 by about 0.00356 while preserving the lower 0.077116 cost ratio, so this checkpoint is kept and becomes the new incumbent.
- **Outcome:** keep
- **Insight:** Continue pivoted trial_0003 from 30M to 45M tokens at K=72; the cheaper-than-baseline recipe reduced latest_exceed enough to reach a new outer best objective of 0.5686420077, beating the retained 0.5721993461 by about 0.00356 while preserving the lower 0.077116 cost ratio, so this checkpoint is kept and becomes the new incumbent.
- **Context:** goal=持续优化 Qwen3-4B layer17 q_proj SAE 部署 objective（latest_exceed 语义）; scope=research/deploy_objective_autoresearch/**,research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/**,checkpoints/qwen3-4B/deploy_objective_autoresearch/**,tests/research/test_deploy_objective_*.py; metric=objective = total_cost_ratio + latest_exceed_alpha_0.50; direction=lower
- **Iteration:** qwen3-4b-layer17-qproj-deploy-objective#13
- **Timestamp:** 2026-04-13T18:55:29Z

### L-6: Continue refined trial_0004 from 30M to 45M tokens at K=72 and LATENTS_PER_EXPERT=104; the wider K72 variant finally cro
- **Strategy:** Continue refined trial_0004 from 30M to 45M tokens at K=72 and LATENTS_PER_EXPERT=104; the wider K72 variant finally crossed the incumbent, lowering the deploy objective to 0.5670499301 and beating the retained 0.5686420077 best by about 0.00159. This checkpoint is kept and becomes the new outer incumbent despite the slightly higher 0.07972 cost ratio.
- **Outcome:** keep
- **Insight:** Continue refined trial_0004 from 30M to 45M tokens at K=72 and LATENTS_PER_EXPERT=104; the wider K72 variant finally crossed the incumbent, lowering the deploy objective to 0.5670499301 and beating the retained 0.5686420077 best by about 0.00159. This checkpoint is kept and becomes the new outer incumbent despite the slightly higher 0.07972 cost ratio.
- **Context:** goal=持续优化 Qwen3-4B layer17 q_proj SAE 部署 objective（latest_exceed 语义）; scope=research/deploy_objective_autoresearch/**,research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/**,checkpoints/qwen3-4B/deploy_objective_autoresearch/**,tests/research/test_deploy_objective_*.py; metric=objective = total_cost_ratio + latest_exceed_alpha_0.50; direction=lower
- **Iteration:** qwen3-4b-layer17-qproj-deploy-objective#18
- **Timestamp:** 2026-04-13T19:32:30Z

### L-7: [PIVOT] Abandon continued structural exploration above the K72 cost frontier after five consecutive non-keep iterations
- **Strategy:** [PIVOT] Abandon continued structural exploration above the K72 cost frontier after five consecutive non-keep iterations since the last keep: the refined K80/LATENTS104 line improved from 0.5810230014 at 15M to 0.5701225933 at 45M, but even its best checkpoint could not beat the retained 0.5670499301 K72/LATENTS104 incumbent and every higher-cost K80/K88 branch has now failed to produce a keep. Switch to a fundamentally different strategy family: keep total_cost_ratio at or below the winning K72 envelope and explore training-hyperparameter changes on that cheaper frontier rather than spending more tokens on larger-K or higher-cost width variants.
- **Outcome:** pivot
- **Insight:** [PIVOT] Abandon continued structural exploration above the K72 cost frontier after five consecutive non-keep iterations since the last keep: the refined K80/LATENTS104 line improved from 0.5810230014 at 15M to 0.5701225933 at 45M, but even its best checkpoint could not beat the retained 0.5670499301 K72/LATENTS104 incumbent and every higher-cost K80/K88 branch has now failed to produce a keep. Switch to a fundamentally different strategy family: keep total_cost_ratio at or below the winning K72 envelope and explore training-hyperparameter changes on that cheaper frontier rather than spending more tokens on larger-K or higher-cost width variants.
- **Context:** goal=持续优化 Qwen3-4B layer17 q_proj SAE 部署 objective（latest_exceed 语义）; scope=research/deploy_objective_autoresearch/**,research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/**,checkpoints/qwen3-4B/deploy_objective_autoresearch/**,tests/research/test_deploy_objective_*.py; metric=objective = total_cost_ratio + latest_exceed_alpha_0.50; direction=lower
- **Iteration:** qwen3-4b-layer17-qproj-deploy-objective#24
- **Timestamp:** 2026-04-13T20:14:57Z

### L-8: Continue pivoted trial_0006 from 30M to 45M tokens at K=64 and LATENTS_PER_EXPERT=104; the cheaper K64 recipe kept its 0
- **Strategy:** Continue pivoted trial_0006 from 30M to 45M tokens at K=64 and LATENTS_PER_EXPERT=104; the cheaper K64 recipe kept its 0.075293 total_cost_ratio while driving latest_exceed_alpha_0.50 down to 0.4897525311, lowering the deploy objective to 0.5650455311. This beats the retained 0.5670499301 K72/LATENTS104 incumbent by about 0.00200, so the outer loop records a keep and promotes the low-cost K64 checkpoint as the new best.
- **Outcome:** keep
- **Insight:** Continue pivoted trial_0006 from 30M to 45M tokens at K=64 and LATENTS_PER_EXPERT=104; the cheaper K64 recipe kept its 0.075293 total_cost_ratio while driving latest_exceed_alpha_0.50 down to 0.4897525311, lowering the deploy objective to 0.5650455311. This beats the retained 0.5670499301 K72/LATENTS104 incumbent by about 0.00200, so the outer loop records a keep and promotes the low-cost K64 checkpoint as the new best.
- **Context:** goal=持续优化 Qwen3-4B layer17 q_proj SAE 部署 objective（latest_exceed 语义）; scope=research/deploy_objective_autoresearch/**,research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/**,checkpoints/qwen3-4B/deploy_objective_autoresearch/**,tests/research/test_deploy_objective_*.py; metric=objective = total_cost_ratio + latest_exceed_alpha_0.50; direction=lower
- **Iteration:** qwen3-4b-layer17-qproj-deploy-objective#27
- **Timestamp:** 2026-04-13T20:42:21Z

### L-9: [PIVOT] Abandon continued width expansion on the low-cost K64 frontier after five consecutive non-keep outer iterations
- **Strategy:** [PIVOT] Abandon continued width expansion on the low-cost K64 frontier after five consecutive non-keep outer iterations since the last keep: the refined K64/LATENTS112 branch improved from 0.5777884003 at 15M to 0.5656414804 at 45M, but even its best checkpoint still could not beat the retained 0.5650455311 K64/LATENTS104 incumbent and it required the higher 0.077897 cost ratio to get there. Switch to a fundamentally different strategy family: keep total_cost_ratio at or below the winning K64 envelope and explore training-hyperparameter changes or same-cost-or-cheaper structure changes instead of adding more width on the same K64 branch.
- **Outcome:** pivot
- **Insight:** [PIVOT] Abandon continued width expansion on the low-cost K64 frontier after five consecutive non-keep outer iterations since the last keep: the refined K64/LATENTS112 branch improved from 0.5777884003 at 15M to 0.5656414804 at 45M, but even its best checkpoint still could not beat the retained 0.5650455311 K64/LATENTS104 incumbent and it required the higher 0.077897 cost ratio to get there. Switch to a fundamentally different strategy family: keep total_cost_ratio at or below the winning K64 envelope and explore training-hyperparameter changes or same-cost-or-cheaper structure changes instead of adding more width on the same K64 branch.
- **Context:** goal=持续优化 Qwen3-4B layer17 q_proj SAE 部署 objective（latest_exceed 语义）; scope=research/deploy_objective_autoresearch/**,research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/**,checkpoints/qwen3-4B/deploy_objective_autoresearch/**,tests/research/test_deploy_objective_*.py; metric=objective = total_cost_ratio + latest_exceed_alpha_0.50; direction=lower
- **Iteration:** qwen3-4b-layer17-qproj-deploy-objective#33
- **Timestamp:** 2026-04-13T21:22:45Z

### L-10: [PIVOT] Abandon further hyperparameter-only tuning around the fixed K64/LATENTS104 recipe after a full post-pivot sweep
- **Strategy:** [PIVOT] Abandon further hyperparameter-only tuning around the fixed K64/LATENTS104 recipe after a full post-pivot sweep failed to produce a keep: lowering LR to 0.0006 still missed by about 0.00269 at 45M, lowering AUXK_ALPHA to 0.015625 came closest but still missed by about 0.00041 at 45M, and collapsing to ACTIVE_EXPERTS=1 cut cost too aggressively and failed badly at 15M. Switch to a fundamentally different structural frontier: keep ACTIVE_EXPERTS at 2 to preserve quality, but lower K from 64 to 56 on the same latents/expert and baseline optimizer settings to test whether a moderate cost reduction can create more objective headroom without the severe reconstruction collapse seen from the one-active-expert recipe.
- **Outcome:** pivot
- **Insight:** [PIVOT] Abandon further hyperparameter-only tuning around the fixed K64/LATENTS104 recipe after a full post-pivot sweep failed to produce a keep: lowering LR to 0.0006 still missed by about 0.00269 at 45M, lowering AUXK_ALPHA to 0.015625 came closest but still missed by about 0.00041 at 45M, and collapsing to ACTIVE_EXPERTS=1 cut cost too aggressively and failed badly at 15M. Switch to a fundamentally different structural frontier: keep ACTIVE_EXPERTS at 2 to preserve quality, but lower K from 64 to 56 on the same latents/expert and baseline optimizer settings to test whether a moderate cost reduction can create more objective headroom without the severe reconstruction collapse seen from the one-active-expert recipe.
- **Context:** goal=持续优化 Qwen3-4B layer17 q_proj SAE 部署 objective（latest_exceed 语义）; scope=research/deploy_objective_autoresearch/**,research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/**,checkpoints/qwen3-4B/deploy_objective_autoresearch/**,tests/research/test_deploy_objective_*.py; metric=objective = total_cost_ratio + latest_exceed_alpha_0.50; direction=lower
- **Iteration:** qwen3-4b-layer17-qproj-deploy-objective#41
- **Timestamp:** 2026-04-13T22:33:58Z

### L-11: Continue refined trial_0011 from 30M to 45M tokens at K=56 and LATENTS_PER_EXPERT=104; the cheaper K56 structure kept it
- **Strategy:** Continue refined trial_0011 from 30M to 45M tokens at K=56 and LATENTS_PER_EXPERT=104; the cheaper K56 structure kept its 0.070866 total_cost_ratio while driving latest_exceed_alpha_0.50 down enough to reach a best deploy objective of 0.5633223766. This beats the retained 0.5650455311 K64/LATENTS104 incumbent by about 0.00172 on a meaningfully cheaper frontier, so the outer loop records a keep and promotes the K56 checkpoint as the new best.
- **Outcome:** keep
- **Insight:** Continue refined trial_0011 from 30M to 45M tokens at K=56 and LATENTS_PER_EXPERT=104; the cheaper K56 structure kept its 0.070866 total_cost_ratio while driving latest_exceed_alpha_0.50 down enough to reach a best deploy objective of 0.5633223766. This beats the retained 0.5650455311 K64/LATENTS104 incumbent by about 0.00172 on a meaningfully cheaper frontier, so the outer loop records a keep and promotes the K56 checkpoint as the new best.
- **Context:** goal=持续优化 Qwen3-4B layer17 q_proj SAE 部署 objective（latest_exceed 语义）; scope=research/deploy_objective_autoresearch/**,research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/**,checkpoints/qwen3-4B/deploy_objective_autoresearch/**,tests/research/test_deploy_objective_*.py; metric=objective = total_cost_ratio + latest_exceed_alpha_0.50; direction=lower
- **Iteration:** qwen3-4b-layer17-qproj-deploy-objective#44
- **Timestamp:** 2026-04-13T23:02:25Z

### L-12: [PIVOT] Abandon further width sweeps around the K56 frontier after five consecutive non-keep outer iterations since the
- **Strategy:** [PIVOT] Abandon further width sweeps around the K56 frontier after five consecutive non-keep outer iterations since the last keep: continuing the retained K56/LATENTS104 line to 60M regressed, widening to LATENTS112 improved but still missed by about 0.00120 at 45M, and shrinking to LATENTS96 failed badly at 15M despite the lower 0.068262 cost ratio. Switch to a fundamentally different strategy family: keep the winning K56/LATENTS104 structure fixed and explore hyperparameter changes on that cheaper frontier, starting with the lower AUXK_ALPHA line that nearly worked on K64, instead of spending more tokens on K56 width changes.
- **Outcome:** pivot
- **Insight:** [PIVOT] Abandon further width sweeps around the K56 frontier after five consecutive non-keep outer iterations since the last keep: continuing the retained K56/LATENTS104 line to 60M regressed, widening to LATENTS112 improved but still missed by about 0.00120 at 45M, and shrinking to LATENTS96 failed badly at 15M despite the lower 0.068262 cost ratio. Switch to a fundamentally different strategy family: keep the winning K56/LATENTS104 structure fixed and explore hyperparameter changes on that cheaper frontier, starting with the lower AUXK_ALPHA line that nearly worked on K64, instead of spending more tokens on K56 width changes.
- **Context:** goal=持续优化 Qwen3-4B layer17 q_proj SAE 部署 objective（latest_exceed 语义）; scope=research/deploy_objective_autoresearch/**,research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/**,checkpoints/qwen3-4B/deploy_objective_autoresearch/**,tests/research/test_deploy_objective_*.py; metric=objective = total_cost_ratio + latest_exceed_alpha_0.50; direction=lower
- **Iteration:** qwen3-4b-layer17-qproj-deploy-objective#49
- **Timestamp:** 2026-04-13T23:50:37Z

### L-13: Continue refined trial_0014 from 30M to 45M tokens on the retained K=56 and LATENTS_PER_EXPERT=104 structure with lower
- **Strategy:** Continue refined trial_0014 from 30M to 45M tokens on the retained K=56 and LATENTS_PER_EXPERT=104 structure with lower AUXK_ALPHA=0.015625; the cheaper 0.070866 cost frontier kept improving and reached a best deploy objective of 0.5631763452. This beats the retained 0.5633223766 K56/LATENTS104 incumbent by about 0.000146 on the same low-cost frontier, so the outer loop records a keep and promotes the 45M low-AUXK checkpoint as the new best.
- **Outcome:** keep
- **Insight:** Continue refined trial_0014 from 30M to 45M tokens on the retained K=56 and LATENTS_PER_EXPERT=104 structure with lower AUXK_ALPHA=0.015625; the cheaper 0.070866 cost frontier kept improving and reached a best deploy objective of 0.5631763452. This beats the retained 0.5633223766 K56/LATENTS104 incumbent by about 0.000146 on the same low-cost frontier, so the outer loop records a keep and promotes the 45M low-AUXK checkpoint as the new best.
- **Context:** goal=持续优化 Qwen3-4B layer17 q_proj SAE 部署 objective（latest_exceed 语义）; scope=research/deploy_objective_autoresearch/**,research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/**,checkpoints/qwen3-4B/deploy_objective_autoresearch/**,tests/research/test_deploy_objective_*.py; metric=objective = total_cost_ratio + latest_exceed_alpha_0.50; direction=lower
- **Iteration:** qwen3-4b-layer17-qproj-deploy-objective#52
- **Timestamp:** 2026-04-14T00:19:57Z
