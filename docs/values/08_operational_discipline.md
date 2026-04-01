# Operational Discipline

## The Principle

Process is a feature, not overhead. The project uses explicit handover protocols, approval gates between phases, mandatory deployment order, tuning matrices (one variable at a time), and infrastructure cleanup after runs.

Every sprint has designated halting points where development STOPS and waits for review. These are not bureaucratic friction — they are the mechanism by which we catch mistakes early, before they compound into expensive rework.

## Why It Matters

**Without process, speed kills.** Moving fast without checkpoints means you can build the wrong thing faster. An approval gate after the spec phase catches a design mistake before hours of implementation are wasted. An approval gate after implementation catches a correctness issue before deployment begins.

**Tuning matrices prevent confusion.** If you change three things at once and the result improves, you don't know which change caused the improvement. If you change three things and the result gets worse, you don't know which change to revert. One variable at a time, documented deltas, measured results.

**Handover protocols prevent knowledge loss.** Every sprint summary is written as if the next reader has total amnesia. This is not pessimistic — it is realistic. AI instances start fresh every session. Team members go on vacation. Handovers ensure continuity.

**From the Financial Topic Modeling project:** The project enforced several operational rules:
- "HALT after Phase X" gates between EDA → ADRs → implementation → validation
- Codex review checkpoints after each sprint before proceeding to the next
- Mandatory deployment order: `batch → ecs → stepfunctions` (cross-module data source dependencies)
- "Step Functions apply after Batch apply" as mandatory deployment sequence
- "Scale down all cost-driving resources at sprint end unless explicitly told otherwise"
- "Do not tune multiple dimensions blindly. Change one primary variable at a time and document the delta from the previous run."

## How to Apply It

1. **Halting points.** Define explicit points in the workflow where development stops for review:
   - After EDA: present findings, get approval
   - After ADRs and specs: Codex reviews design
   - After implementation: Codex reviews code
   - After validation: present evidence, get approval

2. **Tuning matrices.** When optimizing hyperparameters:
   - Establish a baseline with default parameters
   - Change one parameter at a time
   - Record the result for each configuration
   - Present the matrix to the team for decision

3. **Handover documents.** At the end of each sprint, write a retrospective that includes:
   - What was built (deliverables table)
   - What decisions were made and why
   - What was NOT done and why
   - Confidence levels for each decision
   - Required reading list for the next instance

4. **Infrastructure cleanup.** After each run, confirm that cost-driving resources are scaled down. Document the cleanup in the run report.

## When to Invoke This Value

- When tempted to skip a review gate: "Is this gate protecting me from a category of mistake?"
- When tuning parameters: "Am I changing one variable at a time?"
- When ending a work session: "Have I written a handover that a fresh reader could use?"
- When deploying: "Am I following the mandatory deployment order?"
