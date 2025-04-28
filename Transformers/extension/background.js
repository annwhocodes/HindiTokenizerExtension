chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "tokenizeHindi",
    title: "Tokenize Hindi Text",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "tokenizeHindi" && info.selectionText) {
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["content.js"]
      });
      chrome.tabs.sendMessage(tab.id, {
        action: "tokenize",
        text: info.selectionText
      });
    } catch (error) {
      console.error("Error:", error);
    }
  }
});