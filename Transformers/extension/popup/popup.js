chrome.runtime.onMessage.addListener((request) => {
    if (request.action === "showTokens") {
      document.getElementById("tokens").innerHTML = 
        request.tokens.map(token => `<div class="token">${token}</div>`).join("");
    }
  });