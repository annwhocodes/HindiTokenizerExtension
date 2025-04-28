chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "tokenize") {
    fetch("http://localhost:8000/tokenize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: request.text })
    })
    .then(response => response.json())
    .then(tokens => {
      chrome.storage.local.set({ tokens: tokens }, () => {
        chrome.action.openPopup();
      });
    })
    .catch(error => {
      console.error("Error tokenizing text:", error);
    });
  }
  return true; 
});