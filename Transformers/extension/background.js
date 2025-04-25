chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId === "tokenizeHindi" && info.selectionText) {
      try {
        // Inject content.js into the current tab
        await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          files: ["content.js"]
        });
  
        // Send message to content.js AFTER injection
        chrome.tabs.sendMessage(tab.id, {
          action: "tokenize",
          text: info.selectionText
        });
      } catch (error) {
        console.error("Failed to inject content script:", error);
      }
    }
  });