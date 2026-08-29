function findSearchInput() {
  const active = document.activeElement;

  if (active && (
    active.tagName === "INPUT" ||
    active.tagName === "TEXTAREA" ||
    active.isContentEditable
  )) {
    return active;
  }

  return document.querySelector(
    'input[type="search"], input[name="q"], input[name="query"], textarea, [contenteditable="true"]'
  );
}

function insertText(element, text) {
  element.focus();

  if (element.isContentEditable) {
    document.execCommand("insertText", false, text);
  } else {
    const proto = Object.getPrototypeOf(element);
    const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;

    if (setter) setter.call(element, text);
    else element.value = text;
  }

  element.dispatchEvent(new Event("input", { bubbles: true }));
  element.dispatchEvent(new Event("change", { bubbles: true }));
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== "SIGNFOREST_INSERT_TEXT") return;

  const input = findSearchInput();

  if (!input) {
    sendResponse({ ok: false, error: "Campo de busca não encontrado." });
    return;
  }

  insertText(input, message.text);
  sendResponse({ ok: true });
});
