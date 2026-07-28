// The live log view: appends lines and follows the tail unless scrolled up.

export class LogConsole {
  constructor(node) {
    this.node = node;
  }

  append(lines) {
    const atBottom = this.node.scrollHeight - this.node.scrollTop - this.node.clientHeight < 40;
    for (const text of lines) {
      const line = document.createElement("div");
      if (text.startsWith("$ ")) line.className = "cmd";
      else if (text.startsWith("[webui]")) line.className = "sys";
      line.textContent = text;
      this.node.appendChild(line);
    }
    if (atBottom) this.node.scrollTop = this.node.scrollHeight;
  }

  clear() {
    this.node.innerHTML = "";
  }
}
