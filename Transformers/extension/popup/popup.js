document.addEventListener('DOMContentLoaded', () => {
  chrome.storage.local.get(['tokens'], (result) => {
    const tokens = result.tokens || [];
    const tokenListElement = document.getElementById('tokenList');
    
    if (tokens.length > 0) {
      tokens.forEach(token => {
        const listItem = document.createElement('li');
        listItem.textContent = token;
        tokenListElement.appendChild(listItem);
      });
    } else {
      document.getElementById('noTokens').style.display = 'block';
    }
  });
});