import { html, render } from "https://unpkg.com/lit-html?module";

const fileInput = document.getElementById("shelf-input");
const form = document.getElementById("shelf-form");
const statusMsg = document.getElementById("status-msg");
const fileList = document.getElementById("file-list");
const clusterBtn = document.getElementById("cluster-btn");
const addMoreBtn = document.getElementById("add-more");
const dropZone = document.getElementById("drop-zone");
const shelfAssignments = document.getElementById("shelf-assignments");
const shelfHint = document.getElementById("shelf-hint");
const booksPreview = document.getElementById("books-preview");
const booksHint = document.getElementById("books-hint");
const loadingIndicator = document.getElementById("loading-indicator");

let shelves = [];
let identifiedShelves = [];
let isIdentifying = false;

function setStatus(message, type = "") {
  statusMsg.textContent = message;
  statusMsg.className = type === "error" ? "text-danger" : "text-muted";
}

function renderFiles(files) {
  const count = files?.length || 0;
  const queued = shelves.length;
  const items = [
    html`<li class="list-group-item d-flex justify-content-between align-items-center">
      <span>${count ? "1 new photo ready to identify." : "No photo selected yet."}</span>
    </li>`,
    html`<li class="list-group-item d-flex justify-content-between align-items-center">
      <span>${queued ? `${queued} shelf photo${queued === 1 ? "" : "s"} queued for clustering.` : "No shelves queued yet."}</span>
    </li>`,
  ];
  render(html`${items}`, fileList);
}

function setLoading(active) {
  if (!loadingIndicator) return;
  loadingIndicator.classList.toggle("d-none", !active);
}

function getAuthorText(book) {
  if (!book) return "";
  if (Array.isArray(book.authors)) return book.authors.join(", ");
  if (typeof book.authors === "string") return book.authors;
  if (book.author) return book.author;
  return "";
}

function shelfTemplate(payload) {
  if (!payload || !payload.shelves || !Object.keys(payload.shelves).length) {
    shelfHint.textContent = "No shelf data yet.";
    return html``;
  }

  const shelvesPayload = payload.shelves;
  const shelfKeys = Object.keys(shelvesPayload);
  shelfHint.textContent = `Grouped into ${shelfKeys.length} shelf${shelfKeys.length > 1 ? "es" : ""}.`;

  return html`${shelfKeys.map((shelfId, idx) => {
    const books = shelvesPayload[shelfId] || [];
    return html`
      <div class="card border-0 bg-light-subtle">
        <div class="card-body py-2">
          <div class="d-flex justify-content-between align-items-center mb-1">
            <div class="fw-semibold">Shelf ${idx + 1}</div>
            <small class="text-muted">${books.length} book${books.length === 1 ? "" : "s"}</small>
          </div>
        </div>
        <ul class="list-group list-group-flush">
          ${books.length === 0
            ? html`<li class="list-group-item">No books assigned.</li>`
            : books.map((book) => {
                const titleText = book.title || book.text || "Unknown title";
                const authorText = getAuthorText(book);
                return html`
                  <li class="list-group-item">
                    <div class="fw-semibold">${titleText}</div>
                    ${authorText ? html`<small class="text-muted">${authorText}</small>` : ""}
                  </li>
                `;
              })}
        </ul>
      </div>
    `;
  })}`;
}

function renderShelves(payload) {
  render(shelfTemplate(payload), shelfAssignments);
}

function renderIdentifiedBooks() {
  if (!identifiedShelves.length) {
    booksHint.textContent = "Add a shelf to see the titles we found.";
    render(html``, booksPreview);
    return;
  }
  booksHint.textContent = `${identifiedShelves.length} shelf${identifiedShelves.length === 1 ? "" : "s"} identified.`;
  const cards = identifiedShelves.map((shelf, idx) => {
    const books = shelf.books || [];
    return html`
      <div class="card border-0 bg-light-subtle">
        <div class="card-body py-2">
          <div class="d-flex justify-content-between align-items-center mb-1">
            <div class="fw-semibold">Shelf ${idx + 1}</div>
            <small class="text-muted">${books.length} book${books.length === 1 ? "" : "s"}</small>
          </div>
        </div>
        <ul class="list-group list-group-flush">
          ${books.length === 0
            ? html`<li class="list-group-item">No books identified.</li>`
            : books.map((book) => {
                const titleText = book.title || book.text || "Unknown title";
                const authorText = getAuthorText(book);
                return html`
                  <li class="list-group-item">
                    <div class="fw-semibold">${titleText}</div>
                    ${authorText ? html`<small class="text-muted">${authorText}</small>` : ""}
                  </li>
                `;
              })}
        </ul>
      </div>
    `;
  });
  render(html`${cards}`, booksPreview);
}

addMoreBtn.addEventListener("click", () => fileInput.click());

async function identifyShelf(file) {
  if (!file || isIdentifying) {
    return;
  }

  const formData = new FormData();
  formData.append("files", file);

  isIdentifying = true;
  setLoading(true);
  setStatus("Identifying books on this shelf…");

  try {
    const response = await fetch("/shelves", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
      const detail = payload?.detail || "Identification failed.";
      throw new Error(detail);
    }

    const books = payload.books || [];
    const alreadyQueued = shelves.find((queued) => queued.name === file.name);
    if (!alreadyQueued) {
      shelves.push(file);
    }
    identifiedShelves.push({
      label: file.name || `Shelf ${identifiedShelves.length + 1}`,
      books,
    });
    setStatus(`Found ${books.length} book${books.length === 1 ? "" : "s"} on this shelf. Add another or cluster all shelves.`);
    renderIdentifiedBooks();
    renderFiles(null);
    fileInput.value = "";
  } catch (error) {
    const message = error?.message || "Identification failed.";
    setStatus(message, "error");
  } finally {
    isIdentifying = false;
    setLoading(false);
  }
}

fileInput.addEventListener("change", async (event) => {
  const files = event.target.files;
  if (!files || !files.length) {
    renderFiles(files);
    setStatus("");
    return;
  }
  renderFiles(files);
  await identifyShelf(files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("border-warning", "drag");
    dropZone.classList.remove("border-warning-subtle");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("border-warning", "drag");
    dropZone.classList.add("border-warning-subtle");
  });
});

dropZone.addEventListener("drop", (event) => {
  const dropped = event.dataTransfer.files;
  if (dropped && dropped.length) {
    fileInput.files = dropped;
    renderFiles(dropped);
    identifyShelf(dropped[0]);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const files = fileInput.files;
  if (!files || !files.length) {
    setStatus("Add a shelf photo before identifying.", "error");
    return;
  }
  await identifyShelf(files[0]);
});

clusterBtn.addEventListener("click", async () => {
  if (!shelves.length) {
    setStatus("Add at least one shelf before clustering.", "error");
    return;
  }

  const formData = new FormData();
  shelves.forEach((file) => formData.append("files", file));

  clusterBtn.disabled = true;
  setStatus("Clustering shelves…");

  try {
    const response = await fetch("/shelves", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = payload?.detail || "Clustering failed.";
      throw new Error(detail);
    }
    setStatus(`Clustered ${payload.shelf_count} shelf${payload.shelf_count === 1 ? "" : "s"}.`);
    renderShelves(payload);
  } catch (error) {
    const message = error?.message || "Clustering failed.";
    setStatus(message, "error");
  } finally {
    clusterBtn.disabled = false;
  }
});

renderFiles([]);
renderIdentifiedBooks();
