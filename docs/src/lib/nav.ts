export const navigation = [
  {
    label: "Getting started",
    links: [
      ["Install", "docs/install/"],
      ["Quick start", "docs/quick-start/"],
    ],
  },
  {
    label: "Using hspZ",
    links: [
      ["Exactness knobs", "docs/exactness/"],
      ["GPUs and seeding", "docs/gpus/"],
      ["Outputs", "docs/outputs/"],
    ],
  },
  {
    label: "Algorithm",
    links: [
      ["10-minute tour", "docs/algorithm/tour/"],
      ["A. Input + plan", "docs/algorithm/input-plan/"],
      ["B. Spaced index", "docs/algorithm/spaced-index/"],
      ["C. Hit discovery", "docs/algorithm/hit-discovery/"],
      ["D. Score gate", "docs/algorithm/score-gate/"],
      ["E. Materialize", "docs/algorithm/materialize/"],
      ["F. Entropy", "docs/algorithm/entropy/"],
      ["G. Output", "docs/algorithm/output-path/"],
      ["H. Execution", "docs/algorithm/execution/"],
    ],
  },
  {
    label: "Reference",
    links: [["CLI", "docs/cli/"]],
  },
] as const
