/**
 * RVL Libras - Content Script
 * Injeção de caracteres, espaços e remoção no elemento focado da página ativa.
 */

function getEditableTarget() {
  const active = document.activeElement;

  if (
    active &&
    (active.tagName === "INPUT" ||
      active.tagName === "TEXTAREA" ||
      active.isContentEditable)
  ) {
    return active;
  }

  return document.querySelector(
    'input[type="search"], input[name="q"], input[name="query"], textarea, [contenteditable="true"]'
  );
}

function setNativeValue(element, value) {
  const proto = Object.getPrototypeOf(element);
  const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;

  if (setter) {
    setter.call(element, value);
  } else {
    element.value = value;
  }
}

function fireInputEvents(element, char = null) {
  if (char) {
    element.dispatchEvent(
      new KeyboardEvent("keydown", { key: char, bubbles: true, cancelable: true })
    );
    element.dispatchEvent(
      new KeyboardEvent("keypress", { key: char, bubbles: true, cancelable: true })
    );
  }

  element.dispatchEvent(new Event("input", { bubbles: true }));
  element.dispatchEvent(new Event("change", { bubbles: true }));

  if (char) {
    element.dispatchEvent(
      new KeyboardEvent("keyup", { key: char, bubbles: true, cancelable: true })
    );
  }
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

  fireInputEvents(element, char);
}

function deleteLastChar(element) {
  element.focus();

  if (element.isContentEditable) {
    document.execCommand("delete", false, null);
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

  element.dispatchEvent(
    new KeyboardEvent("keydown", { key: "Backspace", code: "Backspace", keyCode: 8, bubbles: true })
  );
  fireInputEvents(element);
  element.dispatchEvent(
    new KeyboardEvent("keyup", { key: "Backspace", code: "Backspace", keyCode: 8, bubbles: true })
  );
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== "SIGNFOREST_KEY") return false;

  const target = getEditableTarget();

  if (!target) {
    sendResponse({ ok: false, error: "Nenhum campo de texto focado ou encontrado." });
    return true;
  }

  if (message.action === "char" && message.value) {
    appendChar(target, message.value);
    sendResponse({ ok: true });
  } else if (message.action === "backspace") {
    deleteLastChar(target);
    sendResponse({ ok: true });
  } else {
    sendResponse({ ok: false, error: "Ação desconhecida." });
  }

  return true;
});