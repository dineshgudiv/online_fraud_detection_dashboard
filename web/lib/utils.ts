export async function copyToClipboard(text: string) {
  if (!text) {
    return;
  }
  if (navigator?.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
}

function parseFilename(disposition: string | null) {
  if (!disposition) {
    return null;
  }
  const match = disposition.match(/filename=\"?([^\";]+)\"?/i);
  return match?.[1] ?? null;
}

export async function downloadFile(url: string) {
  const resp = await fetch(url, { credentials: "include" });
  if (!resp.ok) {
    const message = await resp.text();
    throw new Error(message || `Download failed (${resp.status})`);
  }
  const blob = await resp.blob();
  const filename = parseFilename(resp.headers.get("content-disposition"));
  const objectUrl = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename ?? "certificate.crt";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(objectUrl);
}
