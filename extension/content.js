function getEditableTarget() {
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

function setNativeValue(element, value) {
  const proto = Object.getPrototypeOf(element);
  const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;

  if (setter) setter.call(element, value);
  else element.value = value;
}

function fireInputEvents(element) {
  element.dispatchEvent(new Event("input", { bubbles: true }));
  element.dispatchEvent(new Event("change", { bubbles: true }));
}

function appendChar(element, char) {
  element.focus();

  if (element.isContentEditable) {
    document.execCommand("insertText", false, char);
    return;
  }

  const start = element.selectionStart ?? element.value.length;
  const end = element.selectionEnd ?? element.value.length;
  const value = element.value ?? "";

  const nextValue = value.slice(0, start) + char + value.slice(end);
  setNativeValue(element, nextValue);

  const caret = start + char.length;
  element.setSelectionRange?.(caret, caret);

  fireInputEvents(element);
}

function deleteLastChar(element) {
  element.focus();

  if (element.isContentEditable) {
    document.execCommand("delete", false);
    return;
  }

  const start = element.selectionStart ?? element.value.length;
  const end = element.selectionEnd ?? element.value.length;
  const value = element.value ?? "";

  if (start !== end) {
    const nextValue = value.slice(0, start) + value.slice(end);
    setNativeValue(element, nextValue);
    element.setSelectionRange?.(start, start);
  } else if (start > 0) {
    const nextValue = value.slice(0, start - 1) + value.slice(start);
    setNativeValue(element, nextValue);
    element.setSelectionRange?.(start - 1, start - 1);
  }

  fireInputEvents(element);
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== "SIGNFOREST_KEY") return;

  const target = getEditableTarget();

  if (!target) {
    sendResponse({ ok: false, error: "Nenhum campo de texto focado ou encontrado." });
    return;
  }

  if (message.action === "char" && message.value) {
    appendChar(target, message.value);
  } else if (message.action === "backspace") {
    deleteLastChar(target);
  } else {
    sendResponse({ ok: false, error: "Ação desconhecida." });
    return;
  }

  sendResponse({ ok: true });
});
