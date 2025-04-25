chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "tokenize") {
      fetch("http://localhost:5000/tokenize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: request.text })
      })
      .then(response => response.json())
      .then(tokens => {
        // Reply to the sender (background script)
        sendResponse({ tokens: tokens });
      });
  
      return true; // Required for async sendResponse
    }
  });