// Renders a feature's field spec into the left panel.
//
// A spec is a list of fields; nesting a few of them in an array puts them on
// one row. Each field is:
//
//   { name, label, type: "select" | "text", source, choices, empty,
//     prefer, value, placeholder, hint }
//
//   source      key into /api/options ("configs", "checkpoints") for a select
//   empty       label of a leading blank option (omit to require a real pick)
//   prefer      regex; preselects the first matching choice on first render
//   value       default for a text field

import { clear, el } from "./dom.js";

function preferred(choices, pattern) {
  if (!pattern) return undefined;
  const re = new RegExp(pattern, "i");
  return choices.find((c) => re.test(c));
}

function buildSelect(spec, options, saved) {
  const node = el("select", { id: `f-${spec.name}` });
  const choices = spec.choices || options[spec.source] || [];
  if (spec.empty !== undefined) node.appendChild(el("option", { value: "" }, spec.empty));
  for (const choice of choices) node.appendChild(el("option", { value: choice }, choice));

  const wanted = saved !== undefined ? saved : preferred(choices, spec.prefer);
  if (wanted !== undefined && [...node.options].some((o) => o.value === wanted)) node.value = wanted;
  return node;
}

function buildInput(spec, saved) {
  return el("input", {
    id: `f-${spec.name}`,
    value: saved !== undefined ? saved : (spec.value ?? ""),
    placeholder: spec.placeholder || "",
  });
}

export function renderForm(container, fields, options, saved = {}) {
  clear(container);
  const inputs = {};

  for (const entry of fields) {
    const group = Array.isArray(entry) ? entry : [entry];
    const row = el("div", { class: group.length > 1 ? "row" : "" });
    for (const spec of group) {
      const cell = el("div", { class: "cell" });
      cell.appendChild(el("label", { for: `f-${spec.name}` }, spec.label));
      const input = spec.type === "select"
        ? buildSelect(spec, options, saved[spec.name])
        : buildInput(spec, saved[spec.name]);
      inputs[spec.name] = input;
      cell.appendChild(input);
      if (spec.hint) cell.appendChild(el("div", { class: "hint" }, spec.hint));
      row.appendChild(cell);
    }
    container.appendChild(row);
  }

  return {
    read() {
      return Object.fromEntries(Object.entries(inputs).map(([name, input]) => [name, input.value]));
    },
  };
}
