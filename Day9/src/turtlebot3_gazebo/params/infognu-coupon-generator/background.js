function activeIcon (tabId) {
    console.log("activeIcon")
    chrome.browserAction.setIcon({
        path: chrome.runtime.getURL("../../images/favicon-32x32.png"),
        tabId: tabId
    })
}
function disableIcon (tabId) {
    console.log("disableIcon")
    chrome.browserAction.setIcon({
        path: chrome.runtime.getURL("../../images/disabled-favicon-32x32.png"),
        tabId: tabId
    })
}

async function getCurrentTab() {
    let queryOptions = { active: true, currentWindow: true };
    let [tab] = await chrome.tabs.query(queryOptions);
    return tab;
}

chrome.tabs.onUpdated.addListener(function (tabId, changeInfo, tab) {
    if (changeInfo.status != "complete") return;
    const regex = /[a-zA-Z]+:\/\/[a-zA-Z]+.udemy.com\/course\/[a-zA-Z\/-]*\/lecture*/
    const isInUdemyCourse = regex.test(tab.url);
    
    if (isInUdemyCourse) {
        activeIcon (tabId)
    } else {
        disableIcon (tabId)
    }
})

chrome.runtime.onMessageExternal.addListener(
    function(request, sender, sendResponse) {
        chrome.storage.sync.set({autoCertificateLoading: !request.value || false});
    }
);