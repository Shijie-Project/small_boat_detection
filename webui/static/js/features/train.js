// Train tab -- pairs with webui/features/train.py

export default {
  name: "train",
  fields: [
    { name: "config", label: "Config (-c)", type: "select", source: "configs", prefer: "AEA" },
    {
      name: "checkpoint",
      label: "Tuning checkpoint (-t, optional)",
      type: "select",
      source: "checkpoints",
      empty: "(none / from scratch)",
    },
    [
      { name: "gpus", label: "GPUs (CUDA_VISIBLE_DEVICES)", value: "0", placeholder: "e.g. 0,1" },
      { name: "seed", label: "Seed", value: "42" },
    ],
    [
      { name: "nproc", label: "nproc_per_node", value: "1" },
      { name: "port", label: "master_port", value: "7789" },
    ],
  ],
};
