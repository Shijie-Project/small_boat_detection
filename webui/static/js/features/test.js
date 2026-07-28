// Test tab -- pairs with webui/features/test.py

export default {
  name: "test",
  fields: [
    { name: "config", label: "Config (-c)", type: "select", source: "configs", prefer: "AEA" },
    {
      name: "checkpoint",
      label: "Checkpoint to evaluate (-r, required)",
      type: "select",
      source: "checkpoints",
      empty: "(select a checkpoint)",
      hint: "Results are written next to the checkpoint.",
    },
    [
      { name: "gpus", label: "GPUs (CUDA_VISIBLE_DEVICES)", value: "0", placeholder: "e.g. 0,1" },
      { name: "seed", label: "Seed", value: "42" },
    ],
    [
      { name: "nproc", label: "nproc_per_node", value: "1" },
      { name: "port", label: "master_port", value: "7778" },
    ],
  ],
};
