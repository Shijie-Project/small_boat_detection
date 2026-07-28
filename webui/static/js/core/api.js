// Thin fetch wrappers: JSON in, JSON out, errors as exceptions.

async function parse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || data.message || response.statusText);
  return data;
}

export async function getJSON(url) {
  return parse(await fetch(url));
}

export async function postJSON(url, body) {
  return parse(await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  }));
}
